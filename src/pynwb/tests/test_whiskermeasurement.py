import datetime
import numpy as np

from pynwb import NWBHDF5IO, NWBFile

from pynwb.testing import TestCase, remove_test_file

from ndx_whisk import WhiskerMeasurementTable

def set_up_nwbfile():
    nwbfile = NWBFile(
        session_description='session_description',
        identifier='identifier',
        session_start_time=datetime.datetime.now(datetime.timezone.utc)
    )

    return nwbfile

def create_whisker_measurement_table():
    """Create tabular data to enter into WhiskerMeasurementTable"""
    
    # initialize data
    data = np.zeros((5,), dtype=[('frame_id', 'int32'), ('whisker_id', 'int16'), ('label', 'int16'), ('face_x', 'int32'), ('face_y', 'int32'), ('length', 'float32'), ('pixel_length', 'int16'), ('score', 'float32'), ('angle', 'float32'), ('curvature', 'float32'), ('follicle_x', 'float32'), ('follicle_y', 'float32'), ('tip_x', 'float32'), ('tip_y', 'float32'), ('chunk_start', 'int32')])
        
    # fill the table
    data['frame_id'] = [0, 0, 0, 0, 0]
    data['whisker_id'] = [0, 1, 2, 3, 4]
    data['label'] = [0, 0, 0, 0, 0]
    data['face_x'] = [-179, -179, -179, -179, -179]
    data['face_y'] = [228, 228, 228, 228, 228]
    data['length'] = [234.326065, 263.253693, 194.976227, 75.144081, 52.869373]
    data['pixel_length'] = [232, 231, 159, 75, 53]
    data['score'] = [1801.120239, 1064.221069, 751.812622, 735.624512, 587.142517]
    data['angle'] = [78.098183, 65.121666, 70.394157, 108.658241, 108.656563]
    data['curvature'] = [-0.000170, 0.000661, 0.002244, 0.003572, 0.007454]
    data['follicle_x'] = [127.943810, 115.231041, 100.219406, 181.951691, 192.979843]
    data['follicle_y'] = [223.253464, 244.504532, 252.578522, 82.867287, 73.924797]
    data['tip_x'] = [358.986694, 343.834778, 256.842712, 256.000000, 245.000000]
    data['tip_y'] = [258.152008, 373.196899, 366.157288, 74.349388, 67.047562]
    data['chunk_start'] = [0, 0, 0, 0, 0]
    
    # array([(0, 0, 0, -179, 228, 234.32607 , 232, 1801.1202,  78.09818 , -0.00017 , 127.94381 , 223.25346, 358.9867 , 258.152  , 0),
    #     (0, 1, 0, -179, 228, 263.2537  , 231, 1064.2211,  65.121666,  0.000661, 115.23104 , 244.50453, 343.83478, 373.1969 , 0),
    #     (0, 2, 0, -179, 228, 194.97623 , 159,  751.8126,  70.39416 ,  0.002244, 100.219406, 252.57852, 256.8427 , 366.1573 , 0),
    #     (0, 3, 0, -179, 228,  75.14408 ,  75,  735.6245, 108.65824 ,  0.003572, 181.95169 ,  82.86729, 256.     ,  74.34939, 0),
    #     (0, 4, 0, -179, 228,  52.869373,  53,  587.1425, 108.65656 ,  0.007454, 192.97984 ,  73.9248 , 245.     ,  67.04756, 0)],
    #     dtype=[('frame_id', '<i4'), ('whisker_id', '<i2'), ('label', '<i2'), ('face_x', '<i4'), ('face_y', '<i4'), ('length', '<f4'), ('pixel_length', '<i2'), ('score', '<f4'), ('angle', '<f4'), ('curvature', '<f4'), ('follicle_x', '<f4'), ('follicle_y', '<f4'), ('tip_x', '<f4'), ('tip_y', '<f4'), ('chunk_start', '<i4')])
    
    # convert to dictionary (required for DynamicTable.add_row)
    data = {k: data[k] for k in data.dtype.names}

    return data
 
def read_whisker_measurement_table():
    # Read hdf5 file
    from WhiskiWrap.base import read_whiskers_hdf5_summary
    h5_filename='./ExampleFiles/whiskers.hdf5'
    table = read_whiskers_hdf5_summary(h5_filename)
    print(table.head())

    # import pandas
    # import tables

    # with tables.open_file(h5_filename) as fi:
    #   summary = pandas.DataFrame.from_records(fi.root.summary.read())

    # print(summary.head())
    
    return table
   
class TestWhiskerMeasurementConstructor(TestCase):
# self = TestCase

    def setUp(self):
        """Set up an NWB file."""
        self.nwbfile = set_up_nwbfile()

    def test_constructor(self):
        """Test that the constructor for WhiskerMeasurementTable sets values as expected."""
                
        whisker_data = create_whisker_measurement_table()
        
        whisker_meas = WhiskerMeasurementTable(
            name='name',
            description='description'
        )
        
        for i in range(5):
            # DynamicTable.add_row expects 'dict': change type for 'data'
            # data_dict = {k: data[k][i] for k in data.dtype.names}
            whisker_meas.add_row(whisker_data)

        self.assertEqual(whisker_meas.name, 'name')
        self.assertEqual(whisker_meas.description, 'description')
        np.testing.assert_array_equal(whisker_meas.data, whisker_data)
        self.assertEqual(whisker_meas.colnames, ('frame_id', 'whisker_id', 'label', 'face_x', 'face_y', 'length', 'pixel_length', 'score', 'angle', 'curvature', 'follicle_x', 'follicle_y', 'tip_x', 'tip_y', 'chunk_start'))
        
        
class TestWhiskerMeasurementRoundtrip(TestCase):
    """Simple roundtrip test for WhiskerMeasurementTable."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = 'test_wm_roundtrip.nwb'

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """
        Add a WhiskerMeasurementTable to an NWBFile, write it to file, read the file, and test that the WhiskerMeasurementTable from the file matches the original WhiskerMeasurementTable.
        """
        whisker_data = create_whisker_measurement_table()
        
        whisker_meas = WhiskerMeasurementTable(
            name='name',
            description='description'
        )
        
        for i in range(5):
            whisker_meas.add_row(whisker_data)

        self.nwbfile.processing['behavior'].add(whisker_meas)

        with NWBHDF5IO(self.path, mode='w') as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode='r', load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertContainerEqual(whisker_meas, read_nwbfile.processing['behavior']['name'])
            
class TestWhiskerMeasurementOneWayTrip(TestCase):
    """
    Simple one way trip test for WhiskerMeasurementTable.
    Read from file, write to NWB file, read from NWB file, compare.
    """
    
    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = 'test_wm_oneway.nwb'

    def tearDown(self):
        remove_test_file(self.path)

    def test_one_way_trip(self):
        """
        Read from file, write to NWB file, read from NWB file, compare.
        """
        whisker_data = read_whisker_measurement_table()
        
        whisker_meas = WhiskerMeasurementTable(
            name='name',
            description='description'
        )
        
        for i in range(5):
            whisker_meas.add_row(whisker_data)

        self.nwbfile.processing['behavior'].add(whisker_meas)

        with NWBHDF5IO(self.path, mode='w') as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode='r', load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertContainerEqual(whisker_meas, read_nwbfile.processing['behavior']['name'])

import unittest

if __name__ == '__main__':
    unittest.main()
