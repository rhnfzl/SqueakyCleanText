"""Report-first corpus quality and duplicate selection."""
from .types import CorpusRecord, RecordAnalysis, SelectionConfig, SelectionDecision
from .selection import SelectionReport, analyse_records, select_records

__all__ = ['CorpusRecord', 'RecordAnalysis', 'SelectionConfig', 'SelectionDecision',
           'SelectionReport', 'analyse_records', 'select_records']
