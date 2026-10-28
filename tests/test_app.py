"""Exercise the actual Streamlit choice -> detector integration."""
from pathlib import Path
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("feature", ["A", "B", "C", "D", "H", "DCT", "KP", "SIFT", "DCT_BILATERAL"])
def test_dashboard_feature_choice_runs(feature):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app/main.py', default_timeout=30).run()
    app.radio[0].set_value('Gunakan Citra Sintetis Uji').run()
    app.radio[1].set_value(feature).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3


def test_dataset_tab_uses_readme_links():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app/main.py', default_timeout=30).run()
    content = '\n'.join(element.value for element in app.markdown)
    for path in ['tusharchauhan1898/comofod', 'santoshanand10203040/casiav2', 'ma7moud3tef/coverage']:
        assert f'https://www.kaggle.com/datasets/{path}' in content
    assert 'sophatvathana/casiav2' not in content
    assert 'divg23/coverage-dataset' not in content


def test_sift_shows_only_active_controls_and_demo_evaluation():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app/main.py', default_timeout=30).run()
    app.radio[0].set_value('Gunakan Citra Sintetis Uji').run()
    assert app.radio[1].value == 'SIFT'
    assert 'Langkah Stride (s):' not in [slider.label for slider in app.slider]
    assert 'Kuantisasi Fitur (Q):' not in [field.label for field in app.number_input]
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[1].value != '0 pasangan'
    assert 'Evaluasi Mask pada Citra Ini' in [element.value for element in app.subheader]
