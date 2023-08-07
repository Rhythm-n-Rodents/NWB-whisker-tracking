import datetime, os
import numpy as np
import pandas as pd

from pynwb import NWBHDF5IO, NWBFile

from pynwb.testing import TestCase, remove_test_file

from ndx_whisk import WhiskerMeasurementTable

from WhiskiWrap import wfile_io
from WhiskiWrap.mfile_io import MeasurementsTable
from WhiskiWrap.base import read_whiskers_hdf5_summary, index_measurements

def set_up_nwbfile():
    nwbfile = NWBFile(
        session_description='session_description',
        identifier='identifier',
        session_start_time=datetime.datetime.now(datetime.timezone.utc)
    )
    
    return nwbfile

def initialize_whisker_measurement_table():
    """Initialize tabular data to enter into WhiskerMeasurementTable"""
    
    # With NumPy arrays:
    # measurement_data = np.zeros((5,), dtype=[('frame_id', 'uint32'), ('whisker_id', 'uint16'), ('label', 'uint16'), ('face_x', 'int32'), ('face_y', 'int32'), ('length', 'float32'), ('pixel_length', 'uint16'), ('score', 'float32'), ('angle', 'float32'), ('curvature', 'float32'), ('follicle_x', 'float32'), ('follicle_y', 'float32'), ('tip_x', 'float32'), ('tip_y', 'float32'), ('chunk_start', 'uint32')])
    
    # With Pandas DataFrames:
    
    # Define column types
    column_types = {
        'frame_id': 'uint32',
        'whisker_id': 'uint16',
        'label': 'uint16',
        'face_x': 'int32',
        'face_y': 'int32',
        'length': 'float32',
        'pixel_length': 'uint16',
        'score': 'float32',
        'angle': 'float32',
        'curvature': 'float32',
        'follicle_x': 'float32',
        'follicle_y': 'float32',
        'tip_x': 'float32',
        'tip_y': 'float32',
        'chunk_start': 'uint32'
    }
    
    # measurement_data = pd.DataFrame(columns=['frame_id', 'whisker_id', 'label', 'face_x', 'face_y', 'length', 'pixel_length', 'score', 'angle', 'curvature', 'follicle_x', 'follicle_y', 'tip_x', 'tip_y', 'chunk_start'])
    measurement_data = pd.DataFrame(columns=column_types.keys()).astype(column_types)

    return measurement_data, column_types

def create_whisker_measurement_table():
    """Create tabular data to enter into WhiskerMeasurementTable"""
        
    # initialize data
    measurement_data, column_types = initialize_whisker_measurement_table()
    
    # fill the table
    measurement_data['frame_id'] = [0, 0, 0, 0, 0]
    measurement_data['whisker_id'] = [0, 1, 2, 3, 4]
    measurement_data['label'] = [0, 0, 0, 0, 0]
    measurement_data['face_x'] = [-179, -179, -179, -179, -179]
    measurement_data['face_y'] = [228, 228, 228, 228, 228]
    measurement_data['length'] = [234.326065, 263.253693, 194.976227, 75.144081, 52.869373]
    measurement_data['pixel_length'] = [232, 231, 159, 75, 53]
    measurement_data['score'] = [1801.120239, 1064.221069, 751.812622, 735.624512, 587.142517]
    measurement_data['angle'] = [78.098183, 65.121666, 70.394157, 108.658241, 108.656563]
    measurement_data['curvature'] = [-0.000170, 0.000661, 0.002244, 0.003572, 0.007454]
    measurement_data['follicle_x'] = [127.943810, 115.231041, 100.219406, 181.951691, 192.979843]
    measurement_data['follicle_y'] = [223.253464, 244.504532, 252.578522, 82.867287, 73.924797]
    measurement_data['tip_x'] = [358.986694, 343.834778, 256.842712, 256.000000, 245.000000]
    measurement_data['tip_y'] = [258.152008, 373.196899, 366.157288, 74.349388, 67.047562]
    measurement_data['chunk_start'] = [0, 0, 0, 0, 0]
    
    for col, dtype in column_types.items():
        measurement_data[col] = measurement_data[col].astype(dtype)

    # Equivalent to:
    # array([(0, 0, 0, -179, 228, 234.32607 , 232, 1801.1202,  78.09818 , -0.00017 , 127.94381 , 223.25346, 358.9867 , 258.152  , 0),
    #     (0, 1, 0, -179, 228, 263.2537  , 231, 1064.2211,  65.121666,  0.000661, 115.23104 , 244.50453, 343.83478, 373.1969 , 0),
    #     (0, 2, 0, -179, 228, 194.97623 , 159,  751.8126,  70.39416 ,  0.002244, 100.219406, 252.57852, 256.8427 , 366.1573 , 0),
    #     (0, 3, 0, -179, 228,  75.14408 ,  75,  735.6245, 108.65824 ,  0.003572, 181.95169 ,  82.86729, 256.     ,  74.34939, 0),
    #     (0, 4, 0, -179, 228,  52.869373,  53,  587.1425, 108.65656 ,  0.007454, 192.97984 ,  73.9248 , 245.     ,  67.04756, 0)],
    #     dtype=[('frame_id', '<i4'), ('whisker_id', '<i2'), ('label', '<i2'), ('face_x', '<i4'), ('face_y', '<i4'), ('length', '<f4'), ('pixel_length', '<i2'), ('score', '<f4'), ('angle', '<f4'), ('curvature', '<f4'), ('follicle_x', '<f4'), ('follicle_y', '<f4'), ('tip_x', '<f4'), ('tip_y', '<f4'), ('chunk_start', '<i4')])
    
    # convert to dictionary (required for DynamicTable.add_row)
    # measurement_data_dict = {k: measurement_data[k].tolist() for k in measurement_data.columns} -> this loses the data types
    measurement_data_dict = {k: measurement_data[k].to_numpy() for k in measurement_data.columns}

    return measurement_data_dict
 
def read_whisker_measurement_table(filename):
    # if no filename is given, use the hdf5 example file
    if filename is None:
        filename='./example_files/whiskers.hdf5'
    
    if filename.endswith('.whiskers'):
        # Read whisker file
        whiskers = wfile_io.Load_Whiskers(filename)
        
        # check if measurements file exists
        measurements_filename = filename.replace('.whiskers', '.measurements')
        
        if os.path.isfile(measurements_filename):
            # print(measurements_filename)
            M = MeasurementsTable(str(measurements_filename))
            measurements = M.asarray()
            measurements_idx = 0
        else:
            # return in error
            print("No measurements file found")
            return None
            
        # First check whether classify was run on the measurements file, by comparing whisker ids in the first frame of the whiskers dictionary to the whisker ids in the measurements array

        wid_from_trace = np.array(list(whiskers[0].keys())).astype(int)
        initial_frame_measurements = measurements[:len(wid_from_trace)]
        wid_from_measure = initial_frame_measurements[:, 2].astype(int)

        if not np.array_equal(wid_from_trace, wid_from_measure):
            measurements=index_measurements(whiskers,measurements)

        meas_rows = []
        chunk_start = 0
        
        ## Iterate over rows and append to table
        for frame, frame_whiskers in list(whiskers.items()):
            for whisker_id, wseg in list(frame_whiskers.items()):
                # Write to the table
                
                # If using np array (less efficient for adding multiple rows because it creates a new array each time we add a row):
                
                # # Init array
                # data = np.zeros((5,), dtype=[('frame_id', 'uint32'), ('whisker_id', 'uint16'), ('label', 'uint16'), ('face_x', 'int32'), ('face_y', 'int32'), ('length', 'float32'), ('pixel_length', 'uint16'), ('score', 'float32'), ('angle', 'float32'), ('curvature', 'float32'), ('follicle_x', 'float32'), ('follicle_y', 'float32'), ('tip_x', 'float32'), ('tip_y', 'float32'), ('chunk_start', 'uint32')])

                # # row to add
                # new_row = np.array([(0, 1, 2, 3, 4, 5.0, 6, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14)], dtype=data.dtype)

                # # Add  new row
                # data = np.vstack((data, new_row))
                
                # Using Pandas DataFrame (more efficient for adding multiple rows):
                
                        # Create a new row as a dictionary
                new_meas_row = {
                    'frame_id': wseg.time + chunk_start,
                    'whisker_id': wseg.id,
                    'label': 0,
                    'face_x': M._measurements.contents.face_x,
                    'face_y': M._measurements.contents.face_y,
                    'length': measurements[measurements_idx][3],
                    'pixel_length': len(wseg.x),
                    'score': measurements[measurements_idx][4],
                    'angle': measurements[measurements_idx][5],
                    'curvature': measurements[measurements_idx][6],
                    'follicle_x': measurements[measurements_idx][7],
                    'follicle_y': measurements[measurements_idx][8],
                    'tip_x': measurements[measurements_idx][9],
                    'tip_y': measurements[measurements_idx][10],
                    'chunk_start': chunk_start,
                }
                
                measurements_idx += 1
                assert len(wseg.x) == len(wseg.y)
                
                # Append the new row to the DataFrame
                # meas_table = meas_table.append(new_meas_row, ignore_index=True)
                meas_rows.append(new_meas_row)
                
                # Write whisker contour x and y pixel values
                # %TODO: save whisker contours
                # xpixels_vlarray.append(wseg.x)
                # ypixels_vlarray.append(wseg.y)
                
        # no need to initilize the table, just create a new one
        _, column_types = initialize_whisker_measurement_table()
        meas_table = pd.DataFrame(meas_rows).astype(column_types)
        
    elif filename.endswith('.hdf5'):   
        # Read hdf5 file and returns a pandas DataFrame
        meas_table = read_whiskers_hdf5_summary(filename)
        # print(meas_table.head())

        # Convert non-matching table labels
        meas_table.rename(columns={'fid': 'frame_id', 'wid': 'whisker_id'}, inplace=True)
        
        # list labels
        # print(table.columns)
    
        # # Fix data types if necessary    
        # for col in table.columns:
        #     print(col, table[col].dtype)
        #     if col == 'frame_id':
        #         table[col] = table[col].astype('uint32')
        #     elif col == 'whisker_id':
        #         table[col] = table[col].astype('uint16')
            
        # # list data types
        # table.dtypes
        
    # Convert to dictionary (required for DynamicTable.add_row)
    data_dict = {k: meas_table[k].values for k in meas_table.columns}
    
    return data_dict
 
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
            # take only the k-th value from the dictionary for the ith row
            whisker_meas.add_row({k: whisker_data[k][i] for k in whisker_data.keys()})
            
        self.assertEqual(whisker_meas.name, 'name')
        self.assertEqual(whisker_meas.description, 'description')
        self.assertEqual(whisker_meas.colnames, ('frame_id', 'whisker_id', 'tip_x', 'tip_y', 'follicle_x', 'follicle_y', 'angle', 'label', 'pixel_length', 'length', 'score', 'curvature', 'chunk_start', 'face_x', 'face_y'))        
        for key, expected_values in whisker_data.items():
            np.testing.assert_array_equal(whisker_meas[key][:], expected_values)
        
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
            # take only the k-th value from the dictionary for the ith row
            whisker_meas.add_row({k: whisker_data[k][i] for k in whisker_data.keys()})
            
        # Add a ProcessingModule for behavioral data
        behavior_module = self.nwbfile.create_processing_module(
            name="behavior", description="Processed behavioral data"
        )

        self.nwbfile.processing['behavior'].add(whisker_meas)

        with NWBHDF5IO(self.path, mode='w') as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode='r', load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertContainerEqual(whisker_meas, read_nwbfile.processing['behavior']['name'])
            for key, expected_values in whisker_data.items():
                np.testing.assert_array_equal(expected_values, read_nwbfile.processing['behavior']['name'][key][:])
            
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
        Read from whiskers and measurement files, write to NWB file, read from NWB file, compare.
        """
        whisker_data = read_whisker_measurement_table('example_files/test.whiskers')
        
        whisker_meas = WhiskerMeasurementTable(
            name='name',
            description='description'
        )
        
        # only write the first 5 rows
        for i in range(5):
        # for i in range(np.shape(whisker_data['frame_id'])[0]):
            # take only the k-th value from the dictionary for the ith row
            whisker_meas.add_row({k: whisker_data[k][i] for k in whisker_data.keys()})
            
        # Add a ProcessingModule for behavioral data
        behavior_module = self.nwbfile.create_processing_module(
            name="behavior", description="Processed behavioral data"
        )

        self.nwbfile.processing['behavior'].add(whisker_meas)

        with NWBHDF5IO(self.path, mode='w') as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode='r', load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertContainerEqual(whisker_meas, read_nwbfile.processing['behavior']['name'])
            for key, expected_values in whisker_data.items():
                # compare to the first 5 rows in whisker_data
                np.testing.assert_array_equal(expected_values[:5], read_nwbfile.processing['behavior']['name'][key][:])
                # np.testing.assert_array_equal(expected_values, read_nwbfile.processing['behavior']['name'][key][:])

    def test_one_way_trip_hdf5(self):
        """
        Read from hdf5 file, write to NWB file, read from NWB file, compare.
        """
        whisker_data = read_whisker_measurement_table('example_files/whiskers.hdf5')
        
        whisker_meas = WhiskerMeasurementTable(
            name='name',
            description='description'
        )
        
        # only write the first 5 rows (otherwise it takes quite a long time)
        for i in range(5):
        # for i in range(np.shape(whisker_data['frame_id'])[0]):
            # take only the k-th value from the dictionary for the ith row
            whisker_meas.add_row({k: whisker_data[k][i] for k in whisker_data.keys()})
            
        # Add a ProcessingModule for behavioral data
        behavior_module = self.nwbfile.create_processing_module(
            name="behavior", description="Processed behavioral data"
        )

        self.nwbfile.processing['behavior'].add(whisker_meas)

        with NWBHDF5IO(self.path, mode='w') as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode='r', load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertContainerEqual(whisker_meas, read_nwbfile.processing['behavior']['name'])
            for key, expected_values in whisker_data.items():
                # compare to the first 5 rows in whisker_data
                np.testing.assert_array_equal(expected_values[:5], read_nwbfile.processing['behavior']['name'][key][:])
                # np.testing.assert_array_equal(expected_values, read_nwbfile.processing['behavior']['name'][key][:])
