import copy
import logging
import sys
import time
from enum import IntEnum, auto
from typing import Callable, Union, List

from PySide2.QtCore import QRunnable, QThreadPool, QObject, Signal, \
    Slot, QEvent
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QMainWindow, QProgressBar, \
    QMessageBox as MsgB

import file_manager as fm
import processes as prcs

# from processes import ProcessTask, ProcessTaskResult
from my_logger import setup_logger

loader = QUiLoader()

APP_NAME = 'Example App'

logger = setup_logger(__name__, is_main=True)


class ProgressCmd(IntEnum):
    SetMax = auto()
    AddMax = auto()
    Single = auto()
    Step = auto()
    SetValue = auto()
    Complete = auto()


class StatusCmd(IntEnum):
    ShowMessage = auto()
    Reset = auto()


class ThreadSignals(QObject):
    """
    Defines the signals available from the running worker thread
    """

    finished = Signal(object)
    result = Signal(object)
    progress = Signal(int, int)
    status_update = Signal(int, str)
    error = Signal(Exception)
    test = Signal(str)


class ProgressTracker:
    def __init__(self, num_iterations: int, num_of_trackers: int = 1):
        self._num_iterations = num_iterations
        self._num_of_trackers = num_of_trackers
        self._step_value = 100/num_of_trackers/num_iterations
        self._current_value = 0.0

    def iterate(self) -> int:
        prev_value = self._current_value
        self._current_value += self._step_value
        diff_mod = self._current_value//1 - prev_value//1
        return int(diff_mod)


class BoundThread(QRunnable):
    """
    Worker thread
    """

    def __init__(self, iterations: int = None):
        super().__init__()
        self.signals = ThreadSignals()
        self.iterations = iterations
        self.force_stop = False
        self.is_running = False

    @Slot()
    def run(self):
        """
        Your code goes in this function
        """

        self.is_running = True
        try:
            self.signals.status_update.emit(StatusCmd.ShowMessage,
                                            'Busy with bound thread')
            self.signals.progress.emit(ProgressCmd.SetMax, 100)
            prog_tracker = ProgressTracker(num_iterations=self.iterations)
            for i in range(self.iterations):
                result = i * i
                prog_value = prog_tracker.iterate()
                if prog_value:
                    self.signals.progress.emit(ProgressCmd.Step, prog_value)
                if self.force_stop:
                    break
        except Exception as exception:
            self.signals.error.emit(exception)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self)
            self.signals.status_update.emit(StatusCmd.Reset, '')
            self.signals.progress.emit(ProgressCmd.Complete, 0)
            self.is_running = False


class FunctionThread(QRunnable):
    """
    Worker thread
    """

    def __init__(self, func: Callable, func_args=None, func_kwargs=None):
        super().__init__()
        self.signals = ThreadSignals()
        self.func = func
        self.force_stop = False
        self.is_running = False

        if func_args is None:
            func_args = []
        self.func_args = func_args

        if func_kwargs is None:
            func_kwargs = dict()
        self.func_kwargs = func_kwargs

    def get_force_stop(self) -> bool:
        return self.force_stop

    @Slot()
    def run(self):
        """
        Your code goes in this function
        """

        self.is_running = True
        try:
            func_var_names = self.func.__code__.co_varnames
            # result = None
            args = self.func_args  #.append(result)
            kwargs = self.func_kwargs
            if 'signals' in func_var_names:
                kwargs['signals'] = self.signals
            else:
                if 'progress' in func_var_names:
                    kwargs['progress'] = self.signals.progress
                if 'status_update' in func_var_names:
                    kwargs['status_update'] = self.signals.status_update
            self.signals.status_update.emit(StatusCmd.ShowMessage,
                                            'Busy with function thread')
            result = self.func(*args,
                               get_force_stop=self.get_force_stop, **kwargs)

        except Exception as exception:
            self.signals.error.emit(exception)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self)
            self.signals.status_update.emit(StatusCmd.Reset, '')
            self.signals.progress.emit(ProgressCmd.Complete, 0)
            self.is_running = False


def test_function_thread(iterations, *args, signals: ThreadSignals = None,
                         get_force_stop: Callable = None, **kwargs):
    progress = None
    if signals:
        progress = signals.progress
    else:
        for key, value in kwargs.items():
            if key == 'progress':
                progress = value

    if progress:
        progress.emit(ProgressCmd.SetMax, 100)
        prog_tracker = ProgressTracker(num_iterations=iterations)

    result = int()
    for i in range(iterations):
        result = i * i
        if progress:
            prog_value = prog_tracker.iterate()
            if prog_value:
                progress.emit(ProgressCmd.Step, prog_value)
        if get_force_stop():
            break

    if progress:
        progress.emit(ProgressCmd.Complete, 0)

    return result


class ProcessThreadSignals(QObject):
    """
    Defines the signals available from the running worker thread
    """

    finished = Signal(object)
    result = Signal(object)
    progress = Signal(int, int)
    status_update = Signal(int, str)
    error = Signal(tuple)
    test = Signal(str)


class ProcessThread(QRunnable):
    """
    Worker thread
    """

    def __init__(self, num_processes: int = None,
                 tasks: Union[prcs.ProcessTask, List[prcs.ProcessTask]] = None,
                 signals: ProcessThreadSignals = None,
                 task_buffer_size: int = None):
        super().__init__()
        self._processes = []
        self.task_queue = prcs.create_queue()
        self.result_queue = prcs.create_queue()
        self.force_stop = False
        self.is_running = False

        if num_processes:
            # assert type(num_processes) is int, 'Provided num_processes is ' \
            #                                    'not int'
            if type(num_processes) is not int:
                raise TypeError("Provided num_processes must be of type int")
            self.num_processes = num_processes
        else:
            self.num_processes = prcs.get_max_num_processes()

        if not task_buffer_size:
            task_buffer_size = self.num_processes
        self.task_buffer_size = task_buffer_size

        if not signals:
            self.signals = ProcessThreadSignals()
        else:
            # assert type(signals) is ThreadSignals, 'Provided signals wrong ' \
            #                                        'type'
            if type(signals) is not ThreadSignals:
                raise TypeError("Provided signals must be of type "
                                "ThreadSignals")
            self.signals = signals

        self.tasks = []
        if tasks:
            self.add_task(tasks)
        self.results = []

    def add_tasks(self, tasks: Union[prcs.ProcessTask,
                                     List[prcs.ProcessTask]]):
        if type(tasks) is not List:
            tasks = [tasks]
        all_valid = all([type(task) is prcs.ProcessTask for task in tasks])
        # assert all_valid, "At least some provided tasks are not correct type"
        if not all_valid:
            raise TypeError("At least some of provided tasks are not of "
                            "type ProcessTask")
        self.tasks.extend(tasks)

    def add_tasks_from_methods(self, objects: Union[object, List[object]],
                               method_name: str):
        if type(objects) is not list:
            objects = [objects]
        # assert type(method_name) is str, 'Method_name is not str'
        if type(method_name) is not str:
            raise TypeError("Provided method_name must be of type str")

        all_valid = all([hasattr(obj, method_name) for obj in objects])
        # assert all_valid, 'Some or all objects do not have specified method'
        if not all_valid: raise TypeError("Some or all objects do not have "
                                          "the specified method")

        for obj in objects:
            self.tasks.append(prcs.ProcessTask(obj=obj,
                                               method_name=method_name))

    @Slot()
    def run(self, num_processes: int = None):
        """
        Your code goes in this function
        """

        self.is_running = True
        num_active_processes = 0
        try:
            self.results = [None]*len(self.tasks)
            self.signals.status_update.emit(StatusCmd.ShowMessage,
                                            'Busy with generic worker')
            prog_tracker = ProgressTracker(len(self.tasks))
            self.signals.progress.emit(ProgressCmd.SetMax, 100)
            num_init_tasks = len(self.tasks)
            task_uuids = [task.uuid for task in self.tasks]
            # assert num_init_tasks, 'No tasks were provided'
            if not num_init_tasks: raise TypeError("No tasks were provided")

            num_used_processes = self.num_processes
            if num_init_tasks < self.num_processes:
                num_used_processes = num_init_tasks
            num_active_processes = 0
            for _ in range(num_used_processes):
                process = prcs.SingleProcess(task_queue=self.task_queue,
                                             result_queue=self.result_queue)
                self._processes.append(process)
                process.start()
                num_active_processes += 1

            num_task_left = len(self.tasks)
            tasks_todo = copy.copy(self.tasks)

            init_num = num_used_processes + self.task_buffer_size
            rest = len(tasks_todo) - init_num
            if rest < 0:
                init_num += rest

            for _ in range(init_num):
                self.task_queue.put(tasks_todo.pop(0))

            while num_task_left and not self.force_stop:
                try:
                    result = self.result_queue.get(timeout=1)
                except prcs.get_empty_queue_exception():
                    pass
                else:
                    if len(tasks_todo):
                        self.task_queue.put(tasks_todo.pop(0))

                    if type(result) is not prcs.ProcessTaskResult:
                        raise TypeError("Task result is not of type "
                                        "ProcessTaskResult")

                    ind = task_uuids.index(result.task_uuid)
                    # self.tasks[ind].obj = result.new_task_obj
                    self.results[ind] = result
                    self.result_queue.task_done()
                    num_task_left -= 1
                    prog_value = prog_tracker.iterate()
                    if prog_value:
                        self.signals.progress.emit(ProgressCmd.Step,
                                                   prog_value)

        except Exception as exception:
            self.signals.error.emit(exception)
        # else:
        #     self.signals.result.emit(self.tasks)
        finally:
            if self.force_stop:
                while not self.task_queue.empty():
                    self.task_queue.get()
                    self.task_queue.task_done()
                while not self.result_queue.empty():
                    self.result_queue.get()
                    self.result_queue.task_done()
            if len(self._processes):
                for _ in range(num_used_processes):
                    self.task_queue.put(None)
                for _ in range(num_used_processes):
                    if self.result_queue.get() is True:
                        self.result_queue.task_done()
                        num_active_processes -= 1
                self.task_queue.close()
                self.result_queue.close()
                while any([p.is_alive() for p in self._processes]):
                    time.sleep(1)

            self.signals.result.emit(self.results)
            self.signals.finished.emit(self)
            self.signals.status_update.emit(StatusCmd.Reset, '')
            self.signals.progress.emit(ProgressCmd.Complete, 0)
            self.is_running = False


class Test:
    def __init__(self, iterations: int):
        self.has_run = False
        self.iterations = iterations

    def run(self):
        for i in range(self.iterations):
            result = i * i
        self.has_run = True
        return result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = loader.load(fm.path('main_window.ui', fm.FileType.UI))
        self.setWindowTitle(APP_NAME)

        self.ui.installEventFilter(self)

        self.threadpool = QThreadPool()
        logger.info(f"Multithreading with a maximum of "
                    f"{self.threadpool.maxThreadCount()} threads.")
        self.active_threads = []

        self.ui.btnStartBoundThread.pressed.connect(self.start_bound_thread)
        self.ui.btnStartFunctionThread.pressed.connect(
            self.start_function_thread)
        self.ui.btnStartSingleProcessThread.pressed.connect(
            self.start_single_process_thread)
        self.ui.btnStartMutipleProcessThread.pressed.connect(
            self.start_multiple_process_thread)
        self.ui.btnQuit.pressed.connect(self.quit)

        self.ui.progressBar = QProgressBar(self)
        self.ui.progressBar.setMaximum(0)
        self.ui.progressBar.setVisible(False)
        self.ui.statusBar().addPermanentWidget(self.ui.progressBar)

        self._current_progress_state = ProgressCmd.Complete

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.ui and event.type() == QEvent.Close:
            event.ignore()
            self.quit()
            return True
        else:
            return super().eventFilter(watched, event)

    def quit(self):
        if MsgB.question(self, 'Quit?', 'Are you sure you want to quit?'):
            if len(self.active_threads):
                for worker in self.active_threads:
                    worker.force_stop = True
                self.ui.statusBar().showMessage('Waiting for workers to end')
                self.ui.statusBar().repaint()
                while any([thread.is_running for thread
                           in self.active_threads]):
                    time.sleep(0.1)
            self.ui.removeEventFilter(self)
            app.quit()

    @Slot(int)
    @Slot(int, str)
    def update_status(self, cmd: StatusCmd, new_status: str = None):
        if type(cmd) is int:
            cmd = StatusCmd(cmd)

        if cmd is StatusCmd.Reset:
            self.ui.statusBar().clearMessage()
        if cmd is StatusCmd.ShowMessage:
            if new_status is None:
                new_status = ''
            self.ui.statusBar().showMessage(new_status)

        self.ui.statusBar().repaint()
        self.repaint()

    # @Slot(int)
    @Slot(int, int)
    def update_progress(self, cmd: ProgressCmd, value: int = None):
        if type(cmd) is int:
            cmd = ProgressCmd(cmd)

        if cmd is ProgressCmd.Complete:
            self.ui.progressBar.setMaximum(0)
            self.ui.progressBar.setVisible(False)
        elif cmd is ProgressCmd.SetMax:
            self.ui.progressBar.setValue(0)
            self.ui.progressBar.setMaximum(value)
            self.ui.progressBar.setVisible(True)
        elif cmd is ProgressCmd.AddMax:
            current_max = self.ui.progressBar.maximum()
            self.ui.progressBar.setMaximum(current_max + value)
            self.ui.progressBar.setVisible(True)
        elif cmd is ProgressCmd.Step:
            current_value = self.ui.progressBar.value()
            self.ui.progressBar.setValue(current_value + value)
        elif cmd is ProgressCmd.SetValue:
            self.ui.progressBar.setValue(value)
        elif cmd is ProgressCmd.Single:
            self.ui.progressBar.setMaximum(0)
            self.ui.progressBar.setValue(0)
            self.ui.progressBar.setVisible(True)

        self.ui.progressBar.repaint()
        # self.repaint()
        # QApplication.processEvents()

        # self._current_progress_state = cmd

    def setup_worker_signals(self, thread: Union[BoundThread,
                                                 FunctionThread,
                                                 List[FunctionThread]],
                             is_process_thread: bool = None):
        thread.signals.status_update.connect(self.update_status)
        thread.signals.finished.connect(self.thread_finished)
        if not is_process_thread:
            thread.signals.result.connect(self.thread_results)
        else:
            thread.signals.result.connect(self.process_thread_results)
        thread.signals.progress.connect(self.update_progress)
        thread.signals.error.connect(self.thread_error)

    def start_bound_thread(self):
        thread = BoundThread(iterations=1000000)
        self.setup_worker_signals(thread=thread)
        self.threadpool.start(thread)
        self.active_threads.append(thread)

    def start_function_thread(self):
        thread = FunctionThread(func=test_function_thread,
                                func_args=[100000],
                                func_kwargs=dict())
        self.setup_worker_signals(thread=thread)
        self.threadpool.start(thread)
        self.active_threads.append(thread)

    def start_single_process_thread(self):
        prcs_thread = ProcessThread(num_processes=1)
        self.setup_worker_signals(thread=prcs_thread)
        self.process_test = [Test(10000000) for _ in range(4)]
        prcs_thread.add_tasks_from_methods(self.process_test, 'run')
        self.threadpool.start(prcs_thread)
        self.active_threads.append(prcs_thread)

    def start_multiple_process_thread(self):
        prcs_thread = ProcessThread()
        self.setup_worker_signals(thread=prcs_thread, is_process_thread=True)
        self.process_test = [Test(10000000) for _ in range(20)]
        prcs_thread.add_tasks_from_methods(self.process_test, 'run')
        self.threadpool.start(prcs_thread)
        self.active_threads.append(prcs_thread)

    @Slot(object)
    def thread_finished(self, finished_worker: object):
        for worker in self.active_threads:
            if worker is finished_worker:
                self.active_threads.remove(worker)
        # if self.threadpool.activeThreadCount():
        self.update_progress(ProgressCmd.Complete)
        self.update_status(StatusCmd.Reset)

    @Slot()
    def thread_results(self, result):
        logger.info(f"Thread result: {result}")

    @Slot()
    def process_thread_results(self, task_results: prcs.ProcessTaskResult):
        self.process_test = [res.new_task_obj for res in task_results]
        results = [res.task_return for res in task_results]
        logger.info(f"{len(results)} Process thread  results: {results}")

    @Slot(Exception)
    def thread_error(self, exception: Exception):
        try:
            raise exception
        except Exception as exc:
            logger.error('Thread has failed', exc_info=True)
        self.update_progress(ProgressCmd.Complete)
        self.update_status('Thread error')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.ui.show()
    logger.info("App started and Main Window shown")
    app.exec_()
    logger.info("Application closed, busy shutting down")
    logging.shutdown()
    sys.exit()
