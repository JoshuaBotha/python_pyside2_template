import multiprocessing as mp
from typing import List
from uuid import UUID, uuid1


def create_queue() -> mp.JoinableQueue:
    return mp.JoinableQueue()


def get_max_num_processes() -> int:
    return mp.cpu_count()


def locate_uuid(object_list: List[object], wanted_uuid: UUID):
    all_have_uuid = all([hasattr(obj, 'uuid') for obj in object_list])
    assert all_have_uuid, "Not all objects in object_list have uuid's"
    uuid_list = [obj.uuid for obj in object_list]
    if wanted_uuid not in uuid_list:
        return False, None, None
    else:
        uuid_ind = uuid_list.index(wanted_uuid)
        return True, uuid_ind, uuid_list[uuid_ind]


class ProcessTask:
    def __init__(self, obj: object, method_name: str):
        assert hasattr(obj, method_name), "Object does not have provided " \
                                          "method"
        self.uuid = uuid1()
        self.obj = obj
        self.method_name = method_name


class ProcessTaskResult:
    def __init__(self, task_uuid: UUID,
                 task_return,
                 new_task_obj: ProcessTask):
        self.task_uuid = task_uuid
        self.task_return = task_return
        self.new_task_obj = new_task_obj


class SingleProcess(mp.Process):
    def __init__(self, task_queue: mp.JoinableQueue,
                 result_queue: mp.JoinableQueue):
        mp.Process.__init__(self)
        assert type(task_queue) is mp.queues.JoinableQueue, \
            'task provided not correct type'
        assert type(result_queue) is mp.queues.JoinableQueue, \
            'task provided not correct type'
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        try:
            done = False
            while not done:
                task = self.task_queue.get()
                if task is None:
                    done = True
                    self.task_queue.task_done()
                    self.result_queue.put(None)
                else:
                    task_run = getattr(task.obj, task.method_name)
                    task_return = task_run()
                    process_result = ProcessTaskResult(task_uuid=task.uuid,
                                                       task_return=task_return,
                                                       new_task_obj=task)
                    self.result_queue.put(process_result)
        except Exception as e:
            self.result_queue.put(e)
            pass
