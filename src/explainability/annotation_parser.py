"""Parses CAMELYON16 ASAP-format annotation XMLs into level-0 pixel polygons."""
import xml.etree.ElementTree as ET

def parse_annotation_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    for annotation in root.iter("Annotation"):
        coords = []
        for coord in annotation.iter("Coordinate"):
            x = float(coord.attrib["X"])
            y = float(coord.attrib["Y"])
            coords.append((x, y))
        if len(coords) >= 3:
            polygons.append(coords)
    return polygons
