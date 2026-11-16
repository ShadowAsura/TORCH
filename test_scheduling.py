from random import Random
import unittest

from distributed import Faults, POLICIES, Scheduler, Settings, Status, neighborhoods, simulate_distributed
from experiment import make_workload
from simulator import Job, Node, simulate


def gpu_job(name="gpu-job", arrival=0.2):
    return Job(name, arrival, 1, 64, {"A100": 1})


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.a100 = Node("a100", "A100", 1, 80)
        self.l40 = Node("l40", "L40", 1, 48)
        self.settings = Settings(
            policy="periodic", update_interval=0.5, latency=0.05, retry_interval=0.1,
        )

    def test_centralized_baseline_respects_vram(self):
        result = simulate([self.l40, self.a100], [gpu_job()])
        self.assertEqual(result.placements[0].node, self.a100)

    def test_neighborhoods_are_seeded_symmetric_and_partial(self):
        peers = neighborhoods(12, 1, 42)
        self.assertEqual(peers, neighborhoods(12, 1, 42))
        for worker, neighbors in enumerate(peers):
            self.assertIn((worker + 1) % len(peers), neighbors)
            self.assertLess(len(neighbors), len(peers) - 1)
            for peer in neighbors:
                self.assertIn(worker, peers[peer])

    def test_gossip_discards_old_expired_and_out_of_scope_updates(self):
        scheduler = Scheduler(0, (1, 2), Random(42))
        current = Status(1, self.a100, 0, 2, 3)
        scheduler.receive((current,), now=3, max_age=2)
        scheduler.receive((
            Status(1, self.a100, 1, 1, 4),
            Status(2, self.l40, 1, 1, 1),
            Status(3, self.l40, 1, 1, 4),
        ), now=4, max_age=2)
        self.assertEqual(scheduler.view, {1: current})

    def test_stale_state_cannot_double_book_a_gpu(self):
        result, traffic = simulate_distributed(
            [self.a100, self.l40], [gpu_job("first"), gpu_job("second")], self.settings,
        )
        first, second = result.placements
        self.assertEqual(first.node, second.node)
        self.assertGreaterEqual(second.start, first.finish)
        self.assertGreater(traffic.failed_reservations, 0)
        self.assertGreater(traffic.retries, 0)

    def test_worker_crash_restarts_interrupted_job_on_peer(self):
        backup = Node("backup", "A100", 1, 80)
        result, traffic = simulate_distributed(
            [self.a100, backup], [gpu_job()], self.settings,
            Faults(worker=self.a100.name, crash_at=0.4, recover_at=0.9),
        )
        self.assertEqual([placement.node for placement in result.placements], [backup])
        self.assertAlmostEqual(result.interrupted_gpu_seconds, 0.2)
        self.assertEqual(traffic.interrupted_jobs, 1)
        self.assertEqual(traffic.worker_crashes, 1)
        self.assertEqual(traffic.worker_recoveries, 1)

    def test_dropped_status_messages_still_allow_ring_forwarding(self):
        result, traffic = simulate_distributed(
            [self.l40, self.a100], [gpu_job()], Settings(latency=0.05),
            Faults(drop_status_until=0.8),
        )
        placement, = result.placements
        self.assertLess(placement.start, 0.8)
        self.assertEqual(placement.node, self.a100)
        self.assertGreater(traffic.dropped_status_messages, 0)
        self.assertGreater(traffic.messages["status_update"], traffic.dropped_status_messages)
        self.assertGreater(traffic.messages["forward_job"], 0)

    def test_crash_during_reservation_reply_releases_capacity(self):
        result, traffic = simulate_distributed(
            [self.l40, self.a100], [gpu_job()], self.settings,
            Faults(worker=self.a100.name, crash_at=0.275, recover_at=0.4),
        )
        placement, = result.placements
        self.assertEqual(placement.node, self.a100)
        self.assertGreaterEqual(placement.start, 0.4)
        self.assertGreater(traffic.failed_reservations, 0)
        self.assertEqual(traffic.interrupted_jobs, 0)

    def test_permanent_outage_exhausts_retry_budget(self):
        result, traffic = simulate_distributed(
            [self.a100], [gpu_job()], Settings(max_rounds=3), Faults(worker=self.a100.name),
        )
        self.assertEqual(result.rejected, [gpu_job()])
        self.assertEqual(traffic.retries, 2)
        self.assertEqual(traffic.jobs_exhausted, 1)

    def test_comparison_policies_finish_same_seeded_workload(self):
        nodes = [
            Node(f"{kind}-{index}", kind, 2, vram)
            for kind, vram in (("L40", 48), ("A100", 80))
            for index in range(3)
        ]
        jobs = make_workload(42, 25, 2)
        for policy in POLICIES:
            with self.subTest(policy=policy):
                settings = Settings(policy=policy)
                result, traffic = simulate_distributed(nodes, jobs, settings)
                self.assertCountEqual([placement.job.name for placement in result.placements],
                                      [job.name for job in jobs])
                self.assertFalse(result.rejected)
                self.assertEqual(simulate_distributed(nodes, jobs, settings), (result, traffic))


if __name__ == "__main__":
    unittest.main()
