from blocks.extensions import FinishAfter
from nose.tools import assert_equal

from cuboid.extensions.distribute_extensions import DistributeUpdate, DistributeFinish
from cuboid.extensions.distribute_extensions import DistributeWhetlabFinish
from test_extensions import setup_mainloop
from mock import MagicMock

class Worker:
    def __init__(self):
        pass

def test_distribute_update():
    w = Worker()
    w.commit_changes = MagicMock()

    ex = DistributeUpdate(w, every_n_epochs=1)
    m = setup_mainloop(ex)
    m.run()

    w.commit_changes.assert_called_with("1")

def test_distribute_finish():
    w = Worker()
    w.commit_changes = MagicMock()
    w.finish_job= MagicMock()

    ex = DistributeFinish(w)
    m = setup_mainloop(ex)
    m.run()

    w.commit_changes.assert_called_with("finished@ 1")
    w.finish_job.assert_called()

class Experiment:
    def __init__(self):
        pass
class Res:
    def __init__(self, val):
        self.result_id = val
        self._result_id = val

def test_distribute_whetlab_finish():
    w = Worker()
    w.running_job = "runningjob"
    w.commit_changes = MagicMock()
    w.finish_job = MagicMock()
    w.get_running = MagicMock(return_value=[])

    exp = Experiment()
    exp.update_by_result_id = MagicMock()
    exp.suggest = MagicMock(return_value=Res(22))
    exp.pending = MagicMock(return_value=[])

    def rewrite_jobs_from_func(make_jobs_func):
        jobs = make_jobs_func()
        w.next_jobs = jobs

    w.rewrite_jobs_from_func = rewrite_jobs_from_func

    def score_func():
        return 123

    ex = DistributeWhetlabFinish(w, exp, score_func)
    m = setup_mainloop(ex)
    m.run()

    w.commit_changes.assert_called_with("finished@ 1")
    w.finish_job.assert_called()
    exp.update_by_result_id.assert_called_with("runningjob", 123)
    assert_equal(w.next_jobs, ["22"])
