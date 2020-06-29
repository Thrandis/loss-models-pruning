import atexit


_logger = None
_head_logged = False


def prepare_log(log_path, separator='\t', comment='# ', print_to_console=True,
                buffer_size=50):
    global _logger
    _logger = _Logger(log_path, separator, comment, print_to_console,
                      buffer_size)    


def log_comment(string):
    global _logger
    if _logger is None:
        raise RuntimeError('Logger not defined. Call prepare_log() first!')
    _logger.log_comment(string)


def log_head( heads, formatting=None):
    global _logger
    if _logger is None:
        raise RuntimeError('Logger not defined. Call prepare_log() first!')
    global _head_logged
    if _head_logged:
        raise RuntimeError('Head have already been logged!')
    _logger.log_head(heads, formatting)
    _head_logged = True


def log_values(values):
    global _logger
    if _logger is None:
        raise RuntimeError('Logger not defined. Call prepare_log() first!')
    global _head_logged
    if not _head_logged:
        raise RuntimeError('Head have not been logged yet!')
    _logger.log_values(values)


@atexit.register
def close_log():
    if _logger is not None:
        _logger.close()


class _Logger():

    def __init__(self, log_path, separator='\t', comment='# ',
                 print_to_console=True, buffer_size=50):
        self.separator = separator
        self.comment = comment
        self.log_path = log_path
        self.print_to_console = print_to_console
        self.buffer_size = buffer_size
        if self.log_path is not None:
            self.file = open(log_path, mode='a')
        self.to_log = []

    def _write(self):
        self.file.writelines(self.to_log)
        self.file.flush()
        self.to_log = []

    def close(self):
        if self.log_path is not None:
            if len(self.to_log):
                self._write()
            self.file.flush()
            self.file.close()

    def _log(self, string):
        if self.print_to_console:
            print(string)
        self.to_log.append(string + '\n')
        if self.log_path is not None and len(self.to_log) > self.buffer_size:
            self._write()

    def log_comment(self, string):
        self._log(self.comment + string)

    def log_head(self, heads, formatting=None):
        string = self.separator.join(heads)
        self._log(string)
        if formatting is None:
            formatting = '%.8f' * len(heads)
        self.formatting = formatting

    def log_values(self, values):
        formatted_values = [f % v for f, v in zip(self.formatting, values)]
        string = self.separator.join(formatted_values)
        self._log(string)
