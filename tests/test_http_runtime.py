"""
Tests for the HTTP runtime: whole clusters in one process.

Every node gets its own FastAPI app and `HttpRuntime`. A routing transport
delivers each HTTP request to the app named by the URL's host, so nodes talk
over real HTTP requests without opening ports.
"""

import asyncio

import httpx
import pytest

from torchsched.clusters import from_config
from torchsched.runtime.http import SENDER_HEADER, HttpRuntime, create_app

CENTRALIZED = {
    "design": "centralized",
    "workers": [
        {"id": "w-l40", "gpus": [{"gpu_type": "L40", "count": 2, "vram_gb": 48}]},
        {"id": "w-a100", "gpus": [{"gpu_type": "A100", "count": 2, "vram_gb": 80}]},
    ],
}

GOSSIP = {
    "design": "gossip",
    "seed": 3,
    "workers": [
        {"id": f"l40-{i}", "gpus": [{"gpu_type": "L40", "count": 2, "vram_gb": 48}]}
        for i in range(1, 4)
    ]
    + [{"id": "a100-1", "gpus": [{"gpu_type": "A100", "count": 2, "vram_gb": 80}]}],
    "gossip": {"interval": 0.05, "max_staleness": 1.0, "retry_interval": 0.02},
}


class RoutingTransport(httpx.AsyncBaseTransport):
    """
    Sends each request to the ASGI app registered for its host.
    """

    def __init__(self):
        self.apps = {}

    async def handle_async_request(self, request):
        app = self.apps.get(request.url.host)
        if app is None:
            raise httpx.ConnectError(f"no node at {request.url.host}", request=request)
        return await httpx.ASGITransport(app).handle_async_request(request)


class Cluster:
    def __init__(self, config, *, addresses=None, execution_scale=0.01, fault_injection=False):
        self.transport = RoutingTransport()
        self.client = httpx.AsyncClient(transport=self.transport, timeout=5)
        nodes = from_config(config)
        addresses = addresses or {n.node_id: f"http://{n.node_id}" for n in nodes}
        self.runtimes = {
            n.node_id: HttpRuntime(
                n,
                addresses,
                execution_scale=execution_scale,
                client=self.client,
                fault_injection=fault_injection,
            )
            for n in nodes
        }
        for node_id, runtime in self.runtimes.items():
            self.transport.apps[node_id] = create_app(runtime)

    async def __aenter__(self):
        # ASGITransport does not run app lifespans, so start runtimes here.
        for runtime in self.runtimes.values():
            await runtime.start()
        return self

    async def __aexit__(self, *exc):
        for runtime in self.runtimes.values():
            await runtime.close()
        await self.client.aclose()

    async def submit(self, node_id, job):
        return await self.client.post(f"http://{node_id}/jobs", json=job)

    async def observations(self):
        found = []
        for node_id in self.runtimes:
            response = await self.client.get(f"http://{node_id}/observations")
            found += response.json()["observations"]
        return found

    async def wait_for_completions(self, count, timeout=10.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            done = [o for o in await self.observations() if o["type"] == "JobCompleted"]
            if len(done) >= count:
                return done
            if asyncio.get_running_loop().time() > deadline:
                pytest.fail(f"only {len(done)} of {count} jobs completed")
            await asyncio.sleep(0.02)


def job(job_id, gpus=1, vram=24.0, runtimes=None):
    return {
        "job_id": job_id,
        "gpu_count": gpus,
        "min_vram_gb": vram,
        "runtime_by_gpu": runtimes or {"L40": 1.0, "A100": 0.75},
    }


def test_centralized_cluster_runs_jobs_over_http():
    async def scenario():
        async with Cluster(CENTRALIZED) as cluster:
            for i in range(6):
                response = await cluster.submit("coordinator", job(f"j{i}", gpus=1 + i % 2))
                assert response.status_code == 202
            done = await cluster.wait_for_completions(6)
            assert {o["metric"]["job_id"] for o in done} == {f"j{i}" for i in range(6)}

            # Every worker is back to all GPUs free.
            for worker in ("w-l40", "w-a100"):
                status = (await cluster.client.get(f"http://{worker}/status")).json()
                assert status["status"]["free_gpus"] == [2]
            coordinator = (await cluster.client.get("http://coordinator/status")).json()
            assert coordinator["status"] is None

            metrics = (await cluster.client.get("http://coordinator/metrics")).json()
            assert metrics["send_failures"] == {}
            assert metrics["messages_sent"]["ReserveRequest"] >= 6
            assert metrics["bytes_sent"]["ReserveRequest"] > 0
            worker = (await cluster.client.get("http://w-l40/metrics")).json()
            assert worker["messages_received"].get("ReserveRequest", 0) > 0

    asyncio.run(scenario())


def test_gossip_cluster_forwards_jobs_to_a_capable_neighborhood():
    async def scenario():
        async with Cluster(GOSSIP) as cluster:
            await asyncio.sleep(0.15)  # let a few gossip rounds run
            a100_only = {"A100": 0.5}
            for i in range(4):
                # Submitted to an L40 node, but only the A100 node can run it.
                response = await cluster.submit("l40-1", job(f"big{i}", vram=64.0, runtimes=a100_only))
                assert response.status_code == 202
            done = await cluster.wait_for_completions(4)
            assert {o["metric"]["worker"] for o in done} == {"a100-1"}
            observed = await cluster.observations()
            assert any(o["type"] == "JobForwarded" for o in observed)
            pushes = (await cluster.client.get("http://l40-1/metrics")).json()
            assert pushes["messages_sent"].get("StatusPush", 0) > 0

    asyncio.run(scenario())


def test_api_rejects_bad_requests():
    async def scenario():
        async with Cluster(CENTRALIZED) as cluster:
            client = cluster.client
            # Workers in the centralized design do not schedule.
            assert (await cluster.submit("w-l40", job("x"))).status_code == 409
            invalid = {**job("x"), "gpu_count": 0}
            assert (await cluster.submit("coordinator", invalid)).status_code == 422
            missing = {k: v for k, v in job("x").items() if k != "min_vram_gb"}
            assert (await cluster.submit("coordinator", missing)).status_code == 422
            garbage = await client.post(
                "http://w-l40/messages", content=b"nope", headers={SENDER_HEADER: "coordinator"}
            )
            assert garbage.status_code == 400
            anonymous = await client.post("http://w-l40/messages", content=b"{}")
            assert anonymous.status_code == 400
            assert (await client.get("http://w-l40/health")).json() == {
                "node_id": "w-l40",
                "ok": True,
            }

    asyncio.run(scenario())


def test_unreachable_peers_count_as_send_failures():
    async def scenario():
        # The coordinator has no address, so the workers' boot pushes fail.
        addresses = {"w-l40": "http://w-l40", "w-a100": "http://w-a100"}
        async with Cluster(CENTRALIZED, addresses=addresses) as cluster:
            await asyncio.sleep(0.05)
            for worker in ("w-l40", "w-a100"):
                metrics = (await cluster.client.get(f"http://{worker}/metrics")).json()
                assert metrics["send_failures"] == {"StatusPush": 1}

    asyncio.run(scenario())


def test_jobs_lists_what_a_scheduler_holds():
    async def scenario():
        async with Cluster(CENTRALIZED) as cluster:
            await cluster.submit("coordinator", job("j1", runtimes={"L40": 100.0}))
            await asyncio.sleep(0.05)  # submissions are handled asynchronously
            held = (await cluster.client.get("http://coordinator/jobs")).json()
            assert held == {"node_id": "coordinator", "jobs": ["j1"]}
            await cluster.wait_for_completions(1)
            assert (await cluster.client.get("http://coordinator/jobs")).json()["jobs"] == []
            # Workers in the centralized design do not schedule.
            assert (await cluster.client.get("http://w-l40/jobs")).json()["jobs"] == []

    asyncio.run(scenario())


def test_fault_injection_is_off_unless_enabled():
    async def scenario():
        async with Cluster(CENTRALIZED) as cluster:
            response = await cluster.client.put("http://w-l40/faults", json={"losses": []})
            assert response.status_code == 403

    asyncio.run(scenario())


def test_injected_losses_drop_matching_messages():
    async def scenario():
        async with Cluster(CENTRALIZED, fault_injection=True) as cluster:
            client = cluster.client
            bad = await client.put("http://w-l40/faults", json={"losses": [{"kinds": "X"}]})
            assert bad.status_code == 422
            # Workers' status pushes to the coordinator are lost; the job
            # still runs, since reservation replies carry status too.
            await asyncio.sleep(0.05)  # let the boot pushes through first
            faults = {"losses": [{"kinds": ["StatusPush"]}], "seed": 1}
            for worker in ("w-l40", "w-a100"):
                response = await client.put(f"http://{worker}/faults", json=faults)
                assert response.json()["losses"] == [
                    {"start": 0.0, "probability": 1.0, "kinds": ["StatusPush"]}
                ]
            await cluster.submit("coordinator", job("j1"))
            [done] = await cluster.wait_for_completions(1)
            await asyncio.sleep(0.05)  # the push after the job finished
            metrics = (await client.get(f"http://{done['metric']['worker']}/metrics")).json()
            assert metrics["dropped"]["StatusPush"] >= 1
            assert metrics["send_failures"] == {}
            # Cleared again: nothing more is dropped.
            for worker in ("w-l40", "w-a100"):
                await client.put(f"http://{worker}/faults", json={"losses": []})

    asyncio.run(scenario())


def test_observations_can_be_read_incrementally():
    async def scenario():
        async with Cluster(CENTRALIZED) as cluster:
            await cluster.submit("coordinator", job("j1"))
            await cluster.wait_for_completions(1)
            first = (await cluster.client.get("http://coordinator/observations")).json()
            again = (
                await cluster.client.get(f"http://coordinator/observations?since={first['next']}")
            ).json()
            assert again == {"next": first["next"], "observations": []}
            types = [o["type"] for o in first["observations"]]
            assert types[0] == "JobAccepted" and "JobCompleted" in types

    asyncio.run(scenario())
