from collections import OrderedDict

_report = OrderedDict()


def get_report():
    global _report
    return _report


def add_to_report(name, value, overwrite=False):
    global _report
    if not overwrite and _report.get(name) is not None:
        raise ValueError("The report already has a key " + name)
    _report[name] = value
