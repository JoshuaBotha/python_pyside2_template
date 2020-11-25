import sys
from PySide2.QtCore import QRunnable, QObject, QThreadPool, Slot, Signal
from PySide2.QtWidgets import QApplication, QMainWindow


# Signals class
class WorkerSignals(QObject):
    # test_sig = Signal((), (str,), (int,))
    test_sig = Signal((str,), (int,))
    finished = Signal()


# Test worker
class Worker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        self.signals.test_sig.emit(None)  # Trying to call function
        # with no
        # args
        self.signals.test_sig[int].emit(1)  # Call function with int
        self.signals.test_sig[str].emit('a')  # Call function with str
        self.signals.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        self.run_test()

    def run_test(self):
        worker = Worker()
        # worker.signals.test_sig.connect(self.test_func)
        worker.signals.test_sig[int].connect(self.test_func)
        worker.signals.test_sig[str].connect(self.test_func)
        worker.signals.finished.connect(self.test_finished)
        self.threadpool.start(worker)

    # @Slot()
    @Slot(int)
    @Slot(str)
    def test_func(self, arg=None):
        test_result = 'Test reached'
        if arg:
            test_result += f', arg received and is a {type(arg)}.'
        else:
            test_result += ' and no arg received.'
        print(test_result)

    @Slot()
    def test_finished(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
