"""
Data structures and models particular to msslib.
"""
from xml.etree import ElementTree as etree
import re
import os.path
from shapely.geometry import Polygon

def _pct(pct, pct_of):
    return 100 * float(pct_of)/float(pct)

def _f_pct(pct, ct):
    return float(pct) * float(ct)/100

class Prima:
    def __init__(self):
        self.namespace = ""

    def find(self, element, tag):
        return element.find("%s%s" % (self.namespace, tag))

    def findall(self, element, tag):
        return element.findall("%s%s" % (self.namespace, tag))

class PrimaElement(Prima):
    def __init__(self, namespace, element):
        self.element = element
        self.namespace = namespace
        self.coords = self.get_coords()
        self.polygon = Polygon(self.coords)

    def get_coords(self):
        coords = self.find(self.element, 'Coords')
        return [tuple(map(int, pair.split(","))) for pair in coords.attrib['points'].split()]

    def adjusted_coords(self, a):
        return [(c[0] + a[0], c[1] + a[1]) for c in self.coords]
    
    @property
    def centroid(self):
        return self.polygon.centroid

    @property
    def dimensions(self):
        b = self.polygon.bounds
        return (b[2]-b[0], b[3]-b[1])

    def as_slice(self):
        x1, y1, x2, y2 = self.polygon.bounds
        return (slice(int(y1), int(y2)), slice(int(x1), int(x2)))

class PrimaPage(Prima):
    def __init__(self, file_path):
        self.tree = etree.parse(file_path)
        self.root = self.tree.getroot()
        self.namespace = self.get_ns()
        self.page = self.find(self.root, 'Page')
        self.filename = os.path.splitext(os.path.basename(file_path))[0] 

    def get_ns(self):
        m = re.match('\{.*\}', self.root.tag)
        return m.group(0) if m else ''

    @property
    def dimensions(self):
        return (int(self.page.attrib['imageWidth']), 
                int(self.page.attrib['imageHeight']))
    
    @property
    def border(self):
        return PrimaElement(self.namespace, self.find(self.page, 'Border'))

    @property
    def text_regions(self):
        return [PrimaElement(self.namespace, region) for region in self.findall(self.page, 'TextRegion')]

    @property
    def smallest_region(self):
        return min(self.text_regions, key=lambda x: x.polygon.area)

    def as_percentage(self, dim):
        return (_pct(self.dimensions[0],dim[0]), _pct(self.dimensions[1],dim[1]))

    def from_percentage(self, dim):
        return (_f_pct(dim[0],self.dimensions[0]), _f_pct(dim[1],self.dimensions[1]))
