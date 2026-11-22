import json

import pytest

from sct import DocumentElement, DocumentExtractionRouter, ExtractedDocument, extract_document


class Provider:
    def __init__(self, calls, name, outputs, *, format='pdf', result=None, error=None):
        self.capabilities = {format: frozenset(outputs)}
        self.calls = calls
        self.name = name
        self.result = result
        self.error = error

    def extract(self, path):
        self.calls.append((self.name, path))
        if self.error:
            raise self.error
        return self.result


def test_required_output_filters_before_order_and_pin(tmp_path):
    path = tmp_path / 'input.pdf'
    calls = []
    result = ExtractedDocument((DocumentElement('e0', 'one', page=1, bbox=(0, 0, 1, 1)),))
    providers = {
        'quick': Provider(calls, 'quick', {'text'}, result=result),
        'full': Provider(calls, 'full', {'text', 'ocr', 'tables', 'coordinates'}, result=result),
    }
    required = {'text', 'ocr', 'tables', 'coordinates'}
    router = DocumentExtractionRouter(format='pdf', required=required, providers=providers, fallback=False)
    assert extract_document(path, provider=router) is result
    assert calls == [('full', path)]
    with pytest.raises(ValueError, match='Pinned'):
        DocumentExtractionRouter(format='pdf', required=required, providers=providers, fallback=True, pin='quick')
    assert calls == [('full', path)]


@pytest.mark.parametrize(('fallback', 'pin', 'expected'), [
    (False, None, ['first']),
    (True, 'first', ['first']),
    (True, None, ['first', 'last']),
])
def test_fallback_and_strict_pin(tmp_path, fallback, pin, expected):
    path = tmp_path / 'private.pdf'
    calls = []
    result = ExtractedDocument((DocumentElement('e0', 'one'),))
    router = DocumentExtractionRouter(format='pdf', required={'text'}, fallback=fallback, pin=pin, providers={
        'first': Provider(calls, 'first', {'text'}, error=RuntimeError('private fixture detail')),
        'wrong_format': Provider(calls, 'wrong_format', {'text'}, format='json'),
        'last': Provider(calls, 'last', {'text'}, result=result),
    })
    if len(expected) == 2:
        assert extract_document(path, provider=router) is result
    else:
        with pytest.raises(ValueError, match='^Document extraction failed;') as error:
            extract_document(path, provider=router)
        assert 'private' not in str(error.value)
        assert error.value.__suppress_context__
    assert calls == [(name, path) for name in expected]


@pytest.mark.parametrize('invalid', [
    object(),
    ExtractedDocument((DocumentElement('e0', 'no box', page=1),)),
    ExtractedDocument((DocumentElement('e0', 'no page', bbox=(0, 0, 1, 1)),)),
    ExtractedDocument((DocumentElement('e0', 'no area', page=1, bbox=(0, 0, 0, 1)),)),
    ExtractedDocument((DocumentElement('e0', 'no cells', kind='table_cell', page=1, bbox=(0, 0, 1, 1)),)),
])
def test_invalid_output_can_fall_back(tmp_path, invalid):
    calls = []
    path = tmp_path / 'input.pdf'
    outputs = {'text', 'coordinates', 'tables'}
    result = ExtractedDocument((DocumentElement('e0', 'table', kind='table_cell',
        page=1, bbox=(0, 0, 1, 1), table=0, cell=(0, 1, 0, 2)),))
    router = DocumentExtractionRouter(format='pdf', required=outputs, fallback=True, providers={
        'invalid': Provider(calls, 'invalid', outputs, result=invalid),
        'valid': Provider(calls, 'valid', outputs, result=result),
    })
    assert extract_document(path, provider=router) is result
    assert calls == [('invalid', path), ('valid', path)]


def test_another_format_uses_the_same_api_and_source_ranges(tmp_path):
    class JSONContacts:
        capabilities = {'contacts/json': frozenset({'text'})}

        def extract(self, path):
            lines = json.loads(path.read_text())
            return ExtractedDocument(tuple(DocumentElement(f'e{i}', line) for i, line in enumerate(lines)))

    path = tmp_path / 'contacts.data'
    path.write_text(json.dumps(['Café', 'a@example.com']))
    router = DocumentExtractionRouter(format='contacts/json', required={'text'}, fallback=False, pin='contacts',
                                      providers={'contacts': JSONContacts()})
    document = extract_document(path, provider=router)
    assert document.text == 'Café\na@example.com'
    assert document.source_ranges == ((0, 4), (5, 18))


def test_empty_result_and_table_free_page_are_valid(tmp_path):
    calls = []
    path = tmp_path / 'blank.pdf'
    result = ExtractedDocument(())
    router = DocumentExtractionRouter(format='pdf', required={'text', 'tables', 'ocr'}, fallback=True, providers={
        'full': Provider(calls, 'full', {'text', 'tables', 'ocr'}, result=result),
        'unused': Provider(calls, 'unused', {'text', 'tables', 'ocr'}),
    })
    assert extract_document(path, provider=router) is result
    assert calls == [('full', path)]


@pytest.mark.parametrize('options', [
    {'format': ''}, {'format': 'unknown'}, {'required': set()}, {'required': {'text', 'other'}},
    {'fallback': 'yes'}, {'pin': 'unknown'},
])
def test_invalid_policy_fails_before_any_provider_call(options):
    calls = []
    policy = dict(format='pdf', required={'text'}, fallback=False,
                  providers={'provider': Provider(calls, 'provider', {'text'})})
    with pytest.raises(ValueError):
        DocumentExtractionRouter(**(policy | options))
    assert calls == []
