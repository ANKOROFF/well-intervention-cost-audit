"""Stub of the survey price-sheet loader.

The original module reads the operator's confidential cost-survey workbooks and maps
company names; it is withheld. The functions below are the only names the released
code imports. Reconciliation steps that need the survey sheets raise an error here.
"""
import re

ШАПКИ = {}
ШАПКИ_ТЕКСТ = {}


def норм(s):
    s = re.sub('[^0-9a-zа-яё]', '', str(s).lower().replace('ё', 'е'))
    s = re.sub('(тыс)?руб$', '', s)
    return s.replace('тысруб', '')


def собрать_А():
    raise RuntimeError("The survey price sheets are confidential and not part of this repository.")
