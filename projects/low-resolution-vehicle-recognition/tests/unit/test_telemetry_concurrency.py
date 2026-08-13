from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from roadid.telemetry import EventRecorder


def test_concurrent_writers_receive_unique_continuous_sequence_ids() -> None:
    writer_count = 16
    barrier = Barrier(writer_count)
    recorder = EventRecorder("run-concurrent", capacity=writer_count * 2)

    def write(frame_id: int) -> None:
        barrier.wait()
        recorder.start("vehicle_detection", frame_id=frame_id)
        recorder.complete("vehicle_detection", frame_id=frame_id, duration_ms=1.0)

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        list(executor.map(write, range(writer_count)))

    sequence_ids = [event.sequence_id for event in recorder.all_events()]
    assert sequence_ids == list(range(1, writer_count * 2 + 1))
    assert len({event.event_id for event in recorder.all_events()}) == writer_count * 2


def test_two_waiting_subscribers_observe_the_same_event() -> None:
    recorder = EventRecorder("run-subscribers")
    ready = Barrier(3)

    def observe() -> tuple[int, ...]:
        ready.wait()
        return tuple(event.sequence_id for event in recorder.wait_for_events(0, timeout=2).events)

    with ThreadPoolExecutor(max_workers=2) as executor:
        observers = [executor.submit(observe) for _ in range(2)]
        ready.wait()
        recorder.start("source_acquisition")
        results = [observer.result(timeout=3) for observer in observers]

    assert results == [(1,), (1,)]
