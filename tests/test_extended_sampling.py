from tools.prepare_extended import select_runs


def test_extended_windows_never_join_missing_labels_or_overlap():
    labels={str(i):{} for i in list(range(14))+list(range(20,31))+list(range(40,56))}
    runs=select_runs(labels,20)
    assert len(runs)==2
    assert runs[0]==list(range(12)) and runs[1]==list(range(40,52))
    frames=[i for run in runs for start in range(0,10,3) for i in run[start:start+3]]
    assert len(frames)==len(set(frames))
