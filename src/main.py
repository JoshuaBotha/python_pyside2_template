import sys
import time
from typing import Callable, Union, List
import copy

from PySide2.QtCore import QRunnable, QThreadPool, QObject, Signal, \
    Slot, QEvent, QProcess
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QMainWindow, QProgressBar, \
    QMessageBox as MsgB
# import multiprocessing as mp
from enum import IntEnum, auto

import file_manager as fm
import processes as prcs
# from processes import ProcessTask, ProcessTaskResult

loader = QUiLoader()

APP_NAME = 'Example App'


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
    progress = Signal([int], [int, int])
    status_update = Signal([int], [int, str])
    error = Signal(str)
    test = Signal(str)


class BoundThread(QRunnable):
    """
    Worker thread
    """

    def __init__(self, iterations: int = None):
        super().__init__()
        self.signals = ThreadSignals()
        self.iterations = iterations
        self.force_stop = False

    @Slot()
    def run(self):
        """
        Your code goes in this function
        """

        try:
            self.signals.status_update[int, str].\
                emit('Busy with bound thread')
            self.signals.progress[int, int].emit(
                ProgressCmd.SetMax, self.iterations)
            for i in range(self.iterations):
                result = i * i
                self.signals.progress[int].emit(ProgressCmd.Step)
                if self.force_stop:
                    break
        except Exception as e:
            self.signals.error.emit(e)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self)
            self.signals.status_update[int].emit(StatusCmd.Reset)
            self.signals.progress[int].emit(ProgressCmd.Complete)


class FunctionThread(QRunnable):
    """
    Worker thread
    """

    def __init__(self, func: Callable, func_args=None, func_kwargs=None):
        super().__init__()
        self.signals = ThreadSignals()
        self.func = func
        self.force_stop = False

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

        try:
            func_var_names = self.func.__code__.co_varnames
            result = None
            args = self.func_args.append(result)
            kwargs = self.func_kwargs
            if 'signals' in func_var_names:
                kwargs['signals'] = self.signals
            else:
                if 'progress' in func_var_names:
                    kwargs['progress'] = self.signals.progress
                if 'status_update' in func_var_names:
                    kwargs['status_update'] = self.signals.status_update
            self.signals.status_update[int, str].\
                emit('Busy with function thread')
            result = self.func(*args,
                               get_force_stop=self.get_force_stop, **kwargs)

        except Exception as e:
            self.signals.error.emit(e)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self)
            self.signals.status_update[int].emit(StatusCmd.Reset)
            self.signals.progress[int].emit(ProgressCmd.Complete)


def test_function_thread(iterations, result, *args, signals: ThreadSignals = None,
                         get_force_stop: Callable = None, **kwargs):
    progress = None
    if signals:
        progress = signals.progress
    else:
        for key, value in kwargs.items():
            if key == 'progress':
                progress = value

    if progress:
        progress[int, int].emit(ProgressCmd.SetMax, iterations)

    for i in range(iterations):
        result = i * i
        if progress:
            progress[int].emit(ProgressCmd.Step)
        if get_force_stop:
            break

    if progress:
        progress[int].emit(ProgressCmd.Complete)

    print(result)
    return result


class ProcessThreadSignals(QObject):
    """
    Defines the signals available from the running worker thread
    """

    finished = Signal(object)
    result = Signal(object)
    progress = Signal([int], [int, int])
    status_update = Signal([int], [int, str])
    error = Signal(str)
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

        if num_processes:
            assert type(num_processes) is int, 'Provided num_processes is ' \
                                               'not int'
            self.num_processes = num_processes
        else:
            self.num_processes = prcs.get_max_num_processes()

        if not task_buffer_size:
            task_buffer_size = num_processes
        self.task_buffer_size = task_buffer_size

        if not signals:
            self.signals = ProcessThreadSignals()
        else:
            assert type(signals) is ThreadSignals, 'Provided signals wrong ' \
                                                   'type'
            self.signals = signals

        self.tasks = []
        if tasks:
            self.add_task(tasks)
        # self.process = prcs.SingleProcess(task_queue=self.task_queue,
        #                                   result_queue=self.result_queue)

    def add_tasks(self, tasks: Union[prcs.ProcessTask,
                                     List[prcs.ProcessTask]]):
        if type(tasks) is not List:
            tasks = [tasks]
        all_valid = all([type(task) is ProcessTask for task in tasks])
        assert all_valid, "At least some provided tasks are not correct type"
        self.tasks.extend(tasks)

    def add_tasks_from_methods(self, objects: Union[object, List[object]],
                               method_name: str):
        if type(objects) is not list:
            objects = [objects]
        assert type(method_name) is str, 'Method_name is not str'

        all_valid = all([hasattr(obj, method_name) for obj in objects])
        assert all_valid, 'Some or all objects do not have specified method'

        for obj in objects:
            self.tasks.append(prcs.ProcessTask(obj=obj,
                                               method_name=method_name))

    @Slot()
    def run(self, num_processes: int = None):
        """
        Your code goes in this function
        """

        try:
            self.signals.status_update[int, str]. \
                emit(StatusCmd.ShowMessage, 'Busy with generic worker')
            self.signals.progress[int].emit(ProgressCmd.Single)
            num_init_tasks = len(self.tasks)
            assert num_init_tasks, 'No tasks were provided'

            num_used_processes = self.num_processes
            if num_init_tasks < self.num_processes:
                num_used_processes = num_init_tasks
            for _ in range(num_used_processes):
                process = prcs.SingleProcess(task_queue=self.task_queue,
                                             result_queue=self.result_queue)
                self._processes.append(process)
                process.start()

            active_task_counter = 0
            tasks_todo = copy.copy(self.tasks)

            init_num = num_used_processes + self.task_buffer_size
            rest = len(tasks_todo) - init_num
            if rest < 0:
                init_num = num_used_processes - rest

            for _ in range(init_num):
                self.task_queue.put(tasks_todo.pop(0))
                active_task_counter += 1

            while len(tasks_todo) and not self.force_stop:
                if self.force_stop:
                    for proc in self._processes:
                        proc.close()

                result = self.result_queue.get(timeout=1)
                if result is self.result_queue.em

            # process_done = False
            # while self.process.is_alive():
            #     if len(self.tasks):
            #         task = tasks_todo.pop(0)
            #     else:
            #         task = None
            #     self.task_queue.put(task)
            #     process_result = self.result_queue.get()
            #     self.result_queue.task_done()
            #     if process_result is None:
            #         process_done = True
            #     elif type(process_result) is Exception:
            #         raise process_result
        except Exception as e:
            self.signals.error.emit(e)
        else:
            self.signals.result.emit(self.tasks)
        finally:
            self.signals.finished.emit(self)
            self.signals.status_update[int].emit(StatusCmd.Reset)
            self.signals.progress[int].emit(ProgressCmd.Complete)


class Test:
    def __init__(self, value: int):
        self.has_run = False
        self.value = value

    def sqr(self):
        self.has_run = True
        return self.value * self.value


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = loader.load(fm.path('main_window.ui', fm.FileType.UI))
        self.setWindowTitle(APP_NAME)

        self.ui.installEventFilter(self)

        self.threadpool = QThreadPool()
        print("Multithreading with maximum "
              f" {self.threadpool.maxThreadCount()} threads")
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
                while len(self.active_threads) != 0:
                    time.sleep(0.1)
            self.ui.removeEventFilter(self)
            app.quit()

    @Slot(int)
    @Slot(int, str)
    def update_status(self, cmd: StatusCmd, new_status: str = None):
        if cmd is StatusCmd.Reset:
            self.ui.statusBar().clearMessage()
        if cmd is StatusCmd.ShowMessage:
            if new_status is None:
                new_status = ''
            self.ui.statusBar().showMessage(new_status)
        self.ui.statusBar().repaint()
        self.repaint()

    @Slot(int)
    @Slot(int, int)
    def update_progress(self, cmd: ProgressCmd, value: int = None):
        if cmd is not self._current_progress_state:
            if cmd is ProgressCmd.Complete:
                self.ui.progressBar.setMaximum(0)
                self.ui.progressBar.setVisible(False)
            if cmd is ProgressCmd.SetMax:
                self.ui.progressBar.setMaximum(value)
                self.ui.progressBar.setVisible(True)
            if cmd is ProgressCmd.AddMax:
                current_max = self.ui.progressBar.maximum()
                self.ui.progressBar.setMaximum(current_max + value)
                self.ui.progressBar.setVisible(True)
            if cmd is ProgressCmd.Step:
                current_value = self.ui.progressBar.value()
                self.ui.progressBar.setValue(current_value + 1)
            if cmd is ProgressCmd.SetValue:
                self.ui.progressBar.setValue(value)
            if cmd is ProgressCmd.Single:
                self.ui.progressBar.setMaximum(0)
                self.ui.progressBar.setValue(0)
                self.ui.progressBar.setVisible(True)

            self.ui.progressBar.repaint()
            self.repaint()

            self._current_progress_state = cmd

    def setup_worker_signals(self, thread: Union[BoundThread,
                                                 FunctionThread,
                                                 List[FunctionThread]]):
        thread.signals.status_update[int].connect(self.update_status)
        thread.signals.status_update[int, str].connect(
            self.update_status)
        thread.signals.finished.connect(self.thread_finished)
        thread.signals.result.connect(self.thread_result)
        thread.signals.progress[int].connect(self.update_progress)
        thread.signals.progress[int, int].connect(self.update_progress)
        thread.signals.error.connect(self.thread_error)

    def start_bound_thread(self):
        thread = BoundThread(iterations=100000)
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
        prcs_thread = ProcessThread()
        self.setup_worker_signals(thread=prcs_thread)

        # class Test:
        #     def __init__(self, value: int):
        #         self.value = value
        #
        #     def sqr(self):
        #         return self.value * self.value

        self.test_list = [Test(i) for i in range(10)]
        prcs_thread.add_tasks_from_methods(self.test_list, 'sqr')
        print(f'Before thread start {self.threadpool.activeThreadCount()}')
        self.threadpool.start(prcs_thread)
        print(f'After thread start {self.threadpool.activeThreadCount()}')

    def start_multiple_process_thread(self):
        pass

    @Slot(object)
    def thread_finished(self, finished_worker: object):
        print(f'After thread done: {self.threadpool.activeThreadCount()}')
        for worker in self.active_threads:
            if worker is finished_worker:
                self.active_threads.remove(worker)
        if self.threadpool.activeThreadCount():
            self.update_progress(ProgressCmd.Complete)

    @Slot()
    def thread_result(self, result):
        print(result)

    @Slot(str)
    def thread_error(self, error_message: str):
        print(error_message)
        self.update_progress(ProgressCmd.Complete)
        self.update_status('Worker failed')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.ui.show()
    app.exec_()
