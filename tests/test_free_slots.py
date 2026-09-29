"""scripts/free_slots.py: the parser and the fit rule, on a fixture of the 2026-09-29 cluster state.

That evening 13 GPUs were idle while five 2-GPU jobs pended at 20 CPUs each: the nodes with free
GPUs had 0-16 free CPUs. The tool exists so the request is sized to what a free slot can give,
above the loader's floor - these pin that arithmetic and the parser's handling of DOWN nodes and
CPU-only allocations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'scripts'))
import free_slots as fs  # noqa: E402

FIXTURE = """
NodeName=node03 Arch=x86_64 CPUAlloc=58 CPUTot=64 State=MIXED Partitions=a6000_ada,a6000_ada_interactive
   Gres=gpu:4(S:0-1)
   AllocTRES=cpu=58,gres/gpu=1

NodeName=node11 CPUAlloc=96 CPUTot=96 State=ALLOCATED Partitions=a6000,a6000_interactive
   Gres=gpu:4(S:0-1)
   AllocTRES=cpu=96

NodeName=node12 CPUAlloc=54 CPUTot=64 State=MIXED Partitions=a6000,a6000_interactive
   Gres=gpu:4(S:0-1)
   AllocTRES=cpu=54,gres/gpu=1

NodeName=node61 CPUAlloc=24 CPUTot=40 State=MIXED Partitions=rtx8000,rtx8000_interactive
   Gres=gpu:4(S:0-1)
   AllocTRES=cpu=24,gres/gpu=1

NodeName=node62 CPUAlloc=0 CPUTot=40 State=DOWN+DRAIN+NOT_RESPONDING Partitions=rtx8000
   Gres=gpu:4(S:0-1)
   AllocTRES=

NodeName=node21 CPUAlloc=0 CPUTot=64 State=IDLE Partitions=a100,a100_interactive
   Gres=gpu:4(S:0-1)
   AllocTRES=
"""


def test_parser_reads_free_cpus_and_gpus_and_skips_other_partitions():
    nodes = {n['name']: n for n in fs.parse_nodes(FIXTURE)}
    assert set(nodes) == {'node03', 'node11', 'node12', 'node61', 'node62'}   # a100 excluded
    assert (nodes['node11']['cpu_free'], nodes['node11']['gpu_free']) == (0, 4)  # CPU-only tenant
    assert (nodes['node61']['cpu_free'], nodes['node61']['gpu_free']) == (16, 3)
    assert nodes['node62']['usable'] is False and nodes['node62']['gpu_free'] == 4


def test_two_gpu_job_fits_node61_at_eight_per_gpu_not_the_ten_cap():
    node, per_gpu = fs.best_fit(fs.parse_nodes(FIXTURE), gpus=2, cpu_min=5, cpu_max=10)
    assert node['name'] == 'node61' and per_gpu == 8      # 16 free CPUs / 2 GPUs


def test_floor_is_respected_and_down_nodes_never_chosen():
    nodes = fs.parse_nodes(FIXTURE)
    assert fs.best_fit(nodes, gpus=2, cpu_min=9, cpu_max=10) is None   # node61 gives 8, node62 is down
    assert fs.best_fit(nodes, gpus=1, cpu_min=4, cpu_max=10)[0]['name'] == 'node61'  # 16 > node12's 10


def test_adjust_shrinks_only_when_the_request_is_larger_than_the_fit():
    nodes = fs.parse_nodes(FIXTURE)
    flags, note = fs.adjust_flags(['--gres=gpu:2', '--cpus-per-task=20', '--time=25:00:00'], nodes, 5, 10)
    assert flags == ['--gres=gpu:2', '--cpus-per-task=16', '--time=25:00:00'] and 'node61' in note
    flags, _ = fs.adjust_flags(['--gres=gpu:2', '--cpus-per-task=12'], nodes, 5, 10)
    assert flags == ['--gres=gpu:2', '--cpus-per-task=12']     # never enlarged
    flags, note = fs.adjust_flags(['--gres=gpu:2', '--cpus-per-task=20'], nodes, 9, 10)
    assert flags == ['--gres=gpu:2', '--cpus-per-task=20'] and 'keeping' in note
