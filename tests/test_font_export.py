"""Exporting a PSD must not install fonts or write to the Windows registry."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import text_fidelity


def test_export_only_copies_font_and_notice_into_delivery_directory(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Export must not access the registry')
    monkeypatch.setitem(sys.modules, 'winreg', SimpleNamespace(CreateKey=forbidden, HKEY_CURRENT_USER=0))
    profile = tmp_path / 'user-profile'
    monkeypatch.setenv('LOCALAPPDATA', str(profile))
    delivery = tmp_path / 'delivery'
    delivery.mkdir()
    text_fidelity.export_fonts(delivery, {'LayerStudioSansSC-Regular'})
    assert (delivery / 'fonts' / 'LayerStudioSansSC-Regular.ttf').is_file()
    assert (delivery / 'fonts' / 'Sans-OFL.txt').is_file()
    assert not profile.exists(), 'Export must not alter user-installed fonts'
