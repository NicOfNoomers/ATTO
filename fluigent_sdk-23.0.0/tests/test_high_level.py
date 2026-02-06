# -*- coding: utf-8 -*-
"""
Test the user-facing functions and classes of the SDK
"""

import unittest 
from inspect import signature
from Fluigent import SDK


class TestEnums(unittest.TestCase):
    """Test the enum classes"""
        
    def test_values_are_instances_of_class(self):
        enum_classes = [SDK.fgt_ERROR, SDK.fgt_POWER, SDK.fgt_INSTRUMENT_TYPE,
                            SDK.fgt_SENSOR_CALIBRATION, SDK.fgt_SENSOR_TYPE,
                            SDK.fgt_TTL_MODE]
        for enum_class in enum_classes:
            with self.subTest(enum_class = enum_class):
                enum_values = enum_class.get_values()
                for enum_instance in enum_values:
                    self.assertIsInstance(enum_instance, enum_class)
                    self.assertIs(enum_instance, getattr(enum_class, str(enum_instance)))
                    
    def test_values_are_sorted(self):
        enum_classes = [SDK.fgt_ERROR, SDK.fgt_POWER, SDK.fgt_INSTRUMENT_TYPE,
                            SDK.fgt_SENSOR_CALIBRATION, SDK.fgt_SENSOR_TYPE,
                            SDK.fgt_TTL_MODE]
        for enum_class in enum_classes:
            with self.subTest(enum_class = enum_class):
                # Check that list is sorted
                enum_values = enum_class.get_values()
                enum_int_values = [int(e) for e in enum_values]
                self.assertListEqual(list(range(len(enum_values))), enum_int_values)
        
    def test_compare_to_int(self):
        enum_classes = [SDK.fgt_ERROR, SDK.fgt_POWER, SDK.fgt_INSTRUMENT_TYPE,
                            SDK.fgt_SENSOR_CALIBRATION, SDK.fgt_SENSOR_TYPE,
                            SDK.fgt_TTL_MODE]
        for enum_class in enum_classes:
            with self.subTest(enum_class = enum_class):
                # Enums should compare equal to int
                enum_values = enum_class.get_values()
                for i, value in enumerate(enum_values):
                    self.assertEqual(i, value)


class TestFunctions(unittest.TestCase):
    """Test the user facing functions"""
    def test_call_all_functions(self):
        """Call all functions and check that there are no errors"""
        functions = [f for fname, f in SDK.__dict__.items() 
                    if fname.startswith("fgt_") and type(f) == type(SDK.fgt_init)]
        # Channel info to pass as argument
        ll_channel_info = SDK.low_level.fgt_CHANNEL_INFO()
        ll_channel_info.indexID = 10
        pressure_info = SDK.fgt_CHANNEL_INFO(ll_channel_info)
        ll_channel_info.indexID = 20
        sensor_info = SDK.fgt_CHANNEL_INFO(ll_channel_info)
        ll_channel_info.indexID = 30
        ttl_info = SDK.fgt_CHANNEL_INFO(ll_channel_info)
        
        ll_controller_info = SDK.low_level.fgt_CONTROLLER_INFO()
        ll_controller_info.id = 40
        controller_info = SDK.fgt_CHANNEL_INFO(ll_channel_info)
        print("="*80)
        for f in functions:
            with self.subTest(function = f):
                params = signature(f).parameters
                args = [param_name == "unit" and "mbar" 
                        or param_name == "instruments" and [1] 
                        or param_name == "controller_index" and controller_info 
                        or param_name == "pressure_index" and pressure_info 
                        or param_name == "sensor_index" and sensor_info 
                        or param_name == "ttl_index" and ttl_info 
                        or param_name == "controller_index" and controller_info
                        or 0
                        for param_name in list(params.keys())]
                f(*args)
        print("="*80)

    def test_enum_arguments(self):
        """Check that functions take int or the right enum type, but not a
        different enum"""
        with self.subTest(function = SDK.fgt_set_power):
            SDK.fgt_set_power(0, 0)
            SDK.fgt_set_power(0, SDK.fgt_POWER.POWER_OFF)
            self.assertRaises(ValueError, SDK.fgt_set_power, 0, SDK.fgt_ERROR.OK)
            
        with self.subTest(function = SDK.fgt_set_sensorCalibration):
            SDK.fgt_set_sensorCalibration(0, 0)
            SDK.fgt_set_sensorCalibration(0, SDK.fgt_SENSOR_CALIBRATION.H2O)
            self.assertRaises(ValueError, SDK.fgt_set_sensorCalibration, 0, SDK.fgt_ERROR.OK)
            
        with self.subTest(function = SDK.fgt_set_TtlMode):
            SDK.fgt_set_TtlMode(0, 0)
            SDK.fgt_set_TtlMode(0, SDK.fgt_TTL_MODE.DETECT_RISING_EDGE)
            self.assertRaises(ValueError, SDK.fgt_set_TtlMode, 0, SDK.fgt_ERROR.OK)
            
    def test_default_arguments(self):
        """Test functions that have default arguments"""
        with self.subTest(function = SDK.fgt_get_pressure):
            pressure = SDK.fgt_get_pressure(0)
            self.assertIsInstance(pressure, type(0.1))
            pressure, timestamp = SDK.fgt_get_pressure(0, include_timestamp = True)
            self.assertIsInstance(pressure, float)
            self.assertIsInstance(timestamp, int)
            
        with self.subTest(function = SDK.fgt_get_sensorValue):
            sensor_value = SDK.fgt_get_sensorValue(0)
            self.assertIsInstance(sensor_value, type(0.1))
            sensor_value, timestamp = SDK.fgt_get_sensorValue(0, include_timestamp = True)
            self.assertIsInstance(sensor_value, float)
            self.assertIsInstance(timestamp, int)
            
        with self.subTest(function = SDK.fgt_set_sensorCustomScale):
            SDK.fgt_set_sensorCustomScale(0, 0)
            SDK.fgt_set_sensorCustomScale(0, 0, 0)
            SDK.fgt_set_sensorCustomScale(0, 0, 0, 0)
            SDK.fgt_set_sensorCustomScale(0, 0, 0, 0, 0)
            
        with self.subTest(function = SDK.fgt_init):
            SDK.fgt_init()
            SDK.fgt_init([1,2,3])
            
class TestStructures(unittest.TestCase):
    def test_initialization_fills_all_fields(self):
        with self.subTest(struct = SDK.fgt_CONTROLLER_INFO):
            ll_controller_info = SDK.low_level.fgt_CONTROLLER_INFO()
            ll_controller_info.SN = 100
            ll_controller_info.Firmware = 200
            ll_controller_info.id = 300
            ll_controller_info.type = 4
            controller_info = SDK.fgt_CONTROLLER_INFO(ll_controller_info)
            self.assertEqual(100, controller_info.SN)
            self.assertEqual(200, controller_info.Firmware)
            self.assertEqual(300, controller_info.index)
            self.assertEqual(4, controller_info.InstrType)
            self.assertIsInstance(controller_info.InstrType, SDK.fgt_INSTRUMENT_TYPE)
            controller_info_str = str(controller_info)
            self.assertIn("SN:", controller_info_str)
            self.assertIn("Firmware:", controller_info_str)
            self.assertIn("index:", controller_info_str)
            self.assertIn("InstrType:", controller_info_str)
        
        with self.subTest(struct = SDK.fgt_CHANNEL_INFO):
            ll_channel_info = SDK.low_level.fgt_CHANNEL_INFO()
            ll_channel_info.ControllerSN = 100
            ll_channel_info.firmware = 200
            ll_channel_info.DeviceSN = 300
            ll_channel_info.position = 400
            ll_channel_info.index = 500
            ll_channel_info.indexID = 600
            ll_channel_info.type = 1
            channel_info = SDK.fgt_CHANNEL_INFO(ll_channel_info)
            self.assertEqual(100, channel_info.ControllerSN)
            self.assertEqual(200, channel_info.firmware)
            self.assertEqual(300, channel_info.DeviceSN)
            self.assertEqual(400, channel_info.position)
            self.assertEqual(500, channel_info.index)
            self.assertEqual(600, channel_info.indexID)
            self.assertEqual(1, channel_info.InstrType)
            self.assertIsInstance(channel_info.InstrType, SDK.fgt_INSTRUMENT_TYPE)

    def test_repr_prints_all_fields(self):
        with self.subTest(struct = SDK.fgt_CONTROLLER_INFO):
            ll_controller_info = SDK.low_level.fgt_CONTROLLER_INFO()
            controller_info = SDK.fgt_CONTROLLER_INFO(ll_controller_info)
            controller_info_str = str(controller_info)
            self.assertIn("SN:", controller_info_str)
            self.assertIn("Firmware:", controller_info_str)
            self.assertIn("index:", controller_info_str)
            self.assertIn("InstrType:", controller_info_str)
            
        with self.subTest(struct = SDK.fgt_CHANNEL_INFO):
            ll_channel_info = SDK.low_level.fgt_CHANNEL_INFO()
            channel_info = SDK.fgt_CHANNEL_INFO(ll_channel_info)
            channel_info_str = str(channel_info)
            self.assertIn("ControllerSN:", channel_info_str)
            self.assertIn("firmware:", channel_info_str)
            self.assertIn("DeviceSN:", channel_info_str)
            self.assertIn("position:", channel_info_str)
            self.assertIn("index:", channel_info_str)
            self.assertIn("indexID:", channel_info_str)
            self.assertIn("InstrType:", channel_info_str)
            
    def int_returns_usable_index(self):
        with self.subTest(struct = SDK.fgt_CONTROLLER_INFO):
            ll_controller_info = SDK.low_level.fgt_CONTROLLER_INFO()
            ll_controller_info.id = 56789
            controller_info = SDK.fgt_CONTROLLER_INFO(ll_controller_info)
            self.assertEqual(56789, int(controller_info))
        
        with self.subTest(struct = SDK.fgt_CHANNEL_INFO):
            ll_channel_info = SDK.low_level.fgt_CHANNEL_INFO()
            ll_channel_info.indexID = 123456
            channel_info = SDK.fgt_CHANNEL_INFO(ll_channel_info)
            self.assertEqual(123456, int(channel_info))

if __name__ == '__main__':
    unittest.main()