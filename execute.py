'''
.. versionadded:: 2.37
'''
import logging
import asyncio
import numpy as np
import pandas as pd
import si_prefix as si

import dropbot as db
import dropbot.hardware_test
import dropbot.self_test
import dropbot.threshold

from logging_helpers import _L

NAME = 'dropbot_plugin'


async def actuate(proxy, dmf_device, electrode_states, duration_s=0,
                  volume_threshold=0, c_unit_area=None):
    """
    XXX Coroutine XXX

    Actuate electrodes according to specified states.

    Parameters
    ----------
    electrode_states : pandas.Series
    duration_s : float, optional
        If ``volume_threshold`` step option is set, maximum duration before
        timing out.  Otherwise, time to actuate before actuation is
        considered completed.

    c_unit_area : float, optional
        Specific capacitance, i.e., units of $F/mm^2$.

    Returns
    -------
    actuated_electrodes : list
        List of actuated electrode IDs.


    .. versionchanged:: 2.39.0
        Do not save actuation uuid for volume threshold actuations.
    .. versionchanged:: 2.39.0
        Fix actuated area field typo.
    .. versionchanged:: 2.39.0
        Compute actuated area for static (i.e., delay-based) actuations.
    """
    requested_electrodes = electrode_states[electrode_states > 0].index
    requested_channels = dmf_device.channels_by_electrode.loc[requested_electrodes]

    actuated_channels = pd.Series()
    actuated_area = 0

    channels_updated = asyncio.Event()

    def _on_channels_updated(message):
        '''
        Message keys:
            - ``"n"``: number of actuated channel
            - ``"actuated"``: list of actuated channel identifiers.
            - ``"start"``: ms counter before setting shift registers
            - ``"end"``: ms counter after setting shift registers
        '''
        nonlocal actuated_area
        nonlocal actuated_channels

        actuated_channels = message['actuated']

        if actuated_channels:
            actuated_electrodes = dmf_device.actuated_electrodes(actuated_channels).dropna()
            actuated_areas = dmf_device.electrode_areas.loc[actuated_electrodes.values]
            actuated_area = actuated_areas.sum()
        else:
            actuated_area = 0

        area = actuated_area * (1e-3 ** 2)  # m^2 area
        value, pow10 = si.split(np.sqrt(area))
        si_unit = si.SI_PREFIX_UNITS[len(si.SI_PREFIX_UNITS) // 2 + pow10 // 3]
        status = f'actuated electrodes: {actuated_channels} ({value ** 2} {si_unit}m^2)'
        _L().debug(status)
        channels_updated.set()

    proxy.signals.signal('channels-updated').connect(_on_channels_updated)

    threshold_criteria = [duration_s > 0, volume_threshold > 0, len(requested_electrodes) > 0, c_unit_area is not None]
    _L().debug('threshold_criteria: `%s`', threshold_criteria)

    result = {}
    actuated_areas = dmf_device.electrode_areas.loc[requested_electrodes.values]
    actuated_area = actuated_areas.sum()

    if not all(threshold_criteria):
        # ## Case 1: no volume threshold specified.
        #  1. Set control board state of channels according to requested
        #     actuation states.
        #  2. Wait for channels to be actuated.
        actuated_channels = await asyncio.get_running_loop().run_in_executor(
            None,  # Uses the default executor (a thread pool)
            db.threshold.actuate_channels,  # The function to run in a separate thread
            proxy, requested_channels, 5  # Arguments to the function
        )

        #  3. Connect to `capacitance-updated` signal to record capacitance
        #     values measured during the step.
        capacitance_messages = []

        def _on_capacitance_updated(message):
            message['actuated_channels'] = actuated_channels
            message['actuated_area'] = actuated_area
            capacitance_messages.append(message)

        proxy.signals.signal('capacitance-updated').connect(_on_capacitance_updated)
        try:
            await asyncio.sleep(duration_s)
        finally:
            proxy.signals.signal('capacitance-updated').disconnect(_on_capacitance_updated)

    else:
        # ## Case 2: volume threshold specified.
        #
        # A volume threshold has been set for this step.

        # Calculate target capacitance based on actuated area.
        #
        # Note: `app_values['c_liquid']` represents a *specific
        # capacitance*, i.e., has units of $F/mm^2$.
        meters_squared_area = actuated_area * (1e-3 ** 2)  # m^2 area
        # Approximate length of unit side in SI units.
        si_length, pow10 = si.split(np.sqrt(meters_squared_area))
        si_unit = si.SI_PREFIX_UNITS[len(si.SI_PREFIX_UNITS) // 2 +
                                     pow10 // 3]

        target_capacitance = volume_threshold * actuated_area * c_unit_area

        logger = _L()  # use logger with function context
        if logger.getEffectiveLevel() <= logging.DEBUG:
            message = (
                f'target capacitance: {si.si_format(target_capacitance)}F (actuated area: ({si_length ** 2} {si_unit}m^2) actuated channels: {requested_channels})')
            for line in message.splitlines():
                logger.debug(line)
        # Wait for target capacitance to be reached in background thread,
        # timing out if the specified duration is exceeded.
        try:
            dropbot_event = await asyncio.wait_for(
                db.threshold.co_target_capacitance(proxy, requested_channels, target_capacitance, allow_disabled=False,
                                                   timeout=duration_s), duration_s)
            _L().debug('target capacitance reached: `%s`', dropbot_event)
            actuated_channels = dropbot_event['actuated_channels']

            capacitance_messages = dropbot_event['capacitance_updates']
            for capacitance_i in capacitance_messages:
                capacitance_i['actuated_area'] = actuated_area
                capacitance_i.pop('actuation_uuid1', None)

            result['threshold'] = {'target': dropbot_event['target'], 'measured': dropbot_event['new_value'],
                                   'start': dropbot_event['start'], 'end': dropbot_event['end']}
        except asyncio.TimeoutError:
            raise RuntimeError('Timed out waiting for target capacitance.')

    await channels_updated.wait()
    actuated_electrodes = dmf_device.electrodes_by_channel.loc[actuated_channels]

    # Return list of actuated channels (which _may_ be fewer than the
    # number of requested actuated channels if one or more channels is
    # _disabled_).
    result.update({'actuated_electrodes': actuated_electrodes, 'capacitance_messages': capacitance_messages,
                   'actuated_channels': actuated_channels, 'actuated_area': actuated_area})
    return result


def execute(proxy, dmf_device, plugin_kwargs, signals):
    '''
    Parameters
    ----------
    plugin_kwargs : dict
        Plugin settings as JSON serializable dictionary.
    signals : blinker.Namespace
        Signals namespace.
    '''
    def verify_connected(coro):
        async def _wrapped(*args, **kwargs):
            if proxy is None:
                raise RuntimeError('DropBot not connected.')
            return await coro(*args, **kwargs)
        return _wrapped

    @verify_connected
    async def _set_frequency(frequency):
        proxy.frequency = frequency

    @verify_connected
    async def _set_voltage(voltage):
        proxy.voltage = voltage
    @verify_connected
    async def on_actuation_request(electrode_states, duration_s=0):
        '''
        XXX Coroutine XXX

        Actuate electrodes according to specified states.

        Parameters
        ----------
        electrode_states : pandas.Series
        duration_s : float, optional
            If ``volume_threshold`` step option is set, maximum duration before
            timing out.  Otherwise, time to actuate before actuation is
            considered completed.

        Returns
        -------
        actuated_electrodes : list
            List of actuated electrode IDs.
        '''
        try:
            result = await actuate(proxy, dmf_device, electrode_states, duration_s=duration_s,
                                   volume_threshold=volume_threshold, c_unit_area=c_unit_area)
            responses = signals.signal('actuation-completed').send('dropbot_plugin', **result)
            await asyncio.gather(*(r[1] for r in responses))
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.info('on_actuation_request', exc_info=True)
            raise
        return result['actuated_electrodes']

    kwargs = plugin_kwargs.get(NAME, {})
    volume_threshold = kwargs.get('volume_threshold')
    c_unit_area = kwargs.get('c_unit_area')

    signals.signal('set-frequency').connect(_set_frequency, weak=False)
    signals.signal('set-voltage').connect(_set_voltage, weak=False)
    signals.signal('on-actuation-request').connect(on_actuation_request, weak=False)
