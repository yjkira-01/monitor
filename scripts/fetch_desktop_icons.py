"""Vendor a small, pinned set of Microsoft's MIT Fluent icons as WPF vectors."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REVISION = "cd332ec1f90cda2fad16f86754f8da52f1e9ab30"
BASE = f"https://raw.githubusercontent.com/microsoft/fluentui-system-icons/{REVISION}/"
ICONS = {
    "Video": ("Video", "video"), "Pulse": ("Pulse", "pulse"),
    "People": ("People", "people"), "Alert": ("Alert", "alert"),
    "Document": ("Document Text", "document_text"), "Settings": ("Settings", "settings"),
    "Folder": ("Folder Open", "folder_open"), "Pause": ("Pause", "pause"),
    "Play": ("Play", "play"), "Stop": ("Stop", "stop"),
    "Reset": ("Arrow Reset", "arrow_reset"), "Connect": ("Plug Connected", "plug_connected"),
    "Disconnect": ("Plug Disconnected", "plug_disconnected"), "Add": ("Person Add", "person_add"),
    "Edit": ("Edit", "edit"), "Delete": ("Delete", "delete"),
    "Download": ("Arrow Download", "arrow_download"), "Check": ("Checkmark Circle", "checkmark_circle"),
    "Sparkle": ("Sparkle", "sparkle"), "Refresh": ("Arrow Clockwise", "arrow_clockwise"),
    "Save": ("Save", "save"), "Dismiss": ("Dismiss", "dismiss"),
    "Chevron": ("Chevron Down", "chevron_down"), "Search": ("Search", "search"),
    "Heart": ("Heart Pulse", "heart_pulse", "filled"),
}
DEST = ROOT / "desktop/CareMonitor/Assets/FluentIcons"
DEST.mkdir(parents=True, exist_ok=True)


def fetch(entry):
    key, item = entry
    folder, slug, *style = item
    name = f"ic_fluent_{slug}_24_{style[0] if style else 'regular'}.svg"
    source = BASE + quote(f"assets/{folder}/SVG/{name}")
    target = DEST / name
    if not target.exists():
        with urlopen(source, timeout=40) as response:
            data = response.read()
        svg = ET.fromstring(data)
        assert svg.attrib["viewBox"] == "0 0 24 24"
        target.write_bytes(data)
    data = target.read_bytes()
    svg = ET.fromstring(data)
    paths = svg.findall("{http://www.w3.org/2000/svg}path")
    assert paths and len(paths) == len(list(svg)), f"Unsupported SVG elements in {name}"
    assert all("transform" not in p.attrib for p in paths)
    # Nonzero fill is SVG's default; keep all path subfigures in a single geometry.
    geometry = "F1 " + " ".join(p.attrib["d"] for p in paths)
    return key, geometry, {"key": key, "file": name, "source": source, "sha256": sha256(data).hexdigest()}


with ThreadPoolExecutor(max_workers=6) as pool:
    items = list(pool.map(fetch, ICONS.items()))
license_file = DEST / "LICENSE.txt"
with urlopen(BASE + "LICENSE", timeout=40) as response:
    license_file.write_bytes(response.read())
manifest = {"project": "Microsoft Fluent UI System Icons", "license": "MIT", "revision": REVISION,
            "icons": [item[2] for item in items]}
(DEST / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
resources = ET.Element("ResourceDictionary", {
    "xmlns": "http://schemas.microsoft.com/winfx/2006/xaml/presentation",
    "xmlns:x": "http://schemas.microsoft.com/winfx/2006/xaml",
})
for key, geometry, _ in items:
    ET.SubElement(resources, "Geometry", {"x:Key": "Icon" + key}).text = geometry
brand = ET.SubElement(resources, "DrawingImage", {"x:Key": "AppMark"})
image_drawing = ET.SubElement(brand, "DrawingImage.Drawing")
group = ET.SubElement(image_drawing, "DrawingGroup")
tile = ET.SubElement(group, "GeometryDrawing")
brush = ET.SubElement(tile, "GeometryDrawing.Brush")
gradient = ET.SubElement(brush, "LinearGradientBrush", {"StartPoint": "0,0", "EndPoint": "1,1"})
ET.SubElement(gradient, "GradientStop", {"Color": "#3D8A6C", "Offset": "0"})
ET.SubElement(gradient, "GradientStop", {"Color": "#1C5347", "Offset": "1"})
geometry = ET.SubElement(tile, "GeometryDrawing.Geometry")
ET.SubElement(geometry, "RectangleGeometry", {"Rect": "0,0,64,64", "RadiusX": "15", "RadiusY": "15"})
heart_group = ET.SubElement(group, "DrawingGroup")
transform = ET.SubElement(heart_group, "DrawingGroup.Transform")
ET.SubElement(transform, "MatrixTransform", {"Matrix": "1.95,0,0,1.95,8.6,7.2"})
ET.SubElement(heart_group, "GeometryDrawing", {"Brush": "#FCFFFD", "Geometry": "{StaticResource IconHeart}"})
ET.indent(resources, space="  ")
themes = ROOT / "desktop/CareMonitor/Themes"
themes.mkdir(parents=True, exist_ok=True)
(themes / "Icons.xaml").write_text(ET.tostring(resources, encoding="unicode") + "\n", encoding="utf-8")
print(f"Vendored {len(items)} pinned Fluent vectors and license; generated Themes/Icons.xaml.")
