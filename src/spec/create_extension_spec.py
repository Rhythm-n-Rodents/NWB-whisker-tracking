# -*- coding: utf-8 -*-
import os.path

from pynwb.spec import NWBNamespaceBuilder, export_spec, NWBGroupSpec, NWBAttributeSpec, NWBDatasetSpec


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        doc="""NWB extension to store whisker tracking measurements computed with Whisk (Janelia Whisker Tracker) or other video-based whisker tracking methods.""",
        name="""ndx-whisk""",
        version="""0.1.0""",
        author=list(map(str.strip, """Vincent Prevosto""".split(','))),
        contact=list(map(str.strip, """prevosto@mit.edu""".split(',')))
    )

    ns_builder.include_type('DynamicTable', namespace='core')
    ns_builder.include_type('VectorData', namespace='core')

    # see https://pynwb.readthedocs.io/en/latest/extensions.html#extending-nwb for more information
    whisker_meas = NWBGroupSpec(
        neurodata_type_def='WhiskerMeasurementTable',
        neurodata_type_inc='DynamicTable',
        doc=('A table for storing whisker measurements computed with Whisk.'),
            datasets=[
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='frame_id',
                    doc='The frame ID',
                    dtype='uint32'
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='whisker_id',
                    doc='The whisker ID',
                    dtype='uint16'
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='label',
                    doc='The label assigned to the whisker',
                    dtype='uint16',
                    quantity = "?"
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='tip_x',
                    doc='The x coordinate of the whisker tip',
                    dtype='float32'
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='tip_y',
                    doc='The y coordinate of the whisker tip',
                    dtype='float32'
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='follicle_x',
                    doc='The x coordinate of the whisker follicle',
                    dtype='float32'
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='follicle_y',
                    doc='The y coordinate of the whisker follicle',
                    dtype='float32'
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='angle',
                    doc='The angle of the whisker',
                    dtype='float32'
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='pixel_length',
                    doc='The length of the whisker in pixels',
                    dtype='uint16',
                    quantity = "?"
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='length',
                    doc='The length of the whisker in mm',
                    dtype='float32',
                    quantity = "?"
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='score',
                    doc='The score of the whisker',
                    dtype='float32',
                    quantity = "?"
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='curvature',
                    doc='The curvature of the whisker',
                    dtype='float32',
                    quantity = "?"
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='chunk_start',
                    doc='The index of the first frame of the chunk',
                    dtype='uint32',
                    quantity = "?"
                ),  
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='face_x',
                    doc='The x coordinate of the face',
                    dtype='int32',
                    quantity = "?"
                ),
                NWBDatasetSpec(
                    neurodata_type_inc='VectorData',
                    name='face_y',
                    doc='The y coordinate of the face',
                    dtype='int32',
                    quantity = "?"
                ),
            ],
    )

    new_data_types = [whisker_meas]

    # export the spec to yaml files in the spec folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spec'))
    export_spec(ns_builder, new_data_types, output_dir)
    print('Spec files generated. Please make sure to rerun `pip install .` to load the changes.')


if __name__ == '__main__':
    # usage: python create_extension_spec.py
    main()
