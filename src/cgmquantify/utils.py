import pandas as pd
import numpy as np
from typing import Tuple

def cgms2daybyday(df: pd.DataFrame, dt0=None, inter_gap=45, tz="UTC")->Tuple[np.ndarray, np.ndarray, float]:
    """
        Interpolate glucose values on an equally spaced grid from day to day

        Args:
            df (pd.DataFrame): DataFrame with 'id', 'time', and 'gl' columns (subject ID, time, and glucose values).
            dt0: The time frequency for interpolation in minutes, the default will match the CGM meter's frequency
                (e.g. 5 min for Dexcom).
            inter_gap: The maximum allowable gap (in minutes) for interpolation. The values will not be interpolated
                        between the glucose measurements that are more than inter_gap minutes apart.
                        The default value is 45 min.
            tz: The time zone for interpolation. The default value is UTC.

        Returns:
            Tuple: A tuple containing
                    1) Matrix of glucose values with each row corresponding to a new day,
                       and each column corresponding to time
                    2) Vector of dates corresponding to the rows in the matrix
                    3) Time frequency of the resulting grid, in minutes
        """

    # Check if the date-time format is correct
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce', utc=True)
        if tz != "":
            df['time'] = df['time'].dt.tz_convert(tz)
    df = df.dropna(subset=['time', 'gl'])

    # Check that only one subject is in the data frame. If more than one subject, the first subject will be used.
    if df['id'].nunique() > 1:
        first_id = df['id'].unique()[0]
        print(f"The data frame contains more than one subject ID. Only subject {first_id} was used.")
        df = df[df['id']==first_id]

    g = df['gl'].astype(float).values
    tr = pd.to_datetime(df['time'].values)

    # Check and remove any bad time conversions from above
    not_bad_conv = ~pd.isnull(tr) # double negative means these were NOT bad conversions

    if not not_bad_conv.all():
        print(f" When converting time column, {sum(~not_bad_conv)} rows were set to NA. Check to make sure the time zone is correct.")
        g = g[~not_bad_conv]
        tr = tr[~not_bad_conv]

    # Check for time sorting
    time_diffs = np.diff(tr)/np.timedelta64(1, 'm')

    if (time_diffs < 0).any():
        print(f"The times for subject {df['id'].iloc[0]} are not in increasing order. Sorting automatically.")
        sorted_indx = np.argsort(tr)
        tr = tr[sorted_indx]
        g = g[sorted_indx]
        time_diffs = np.diff(tr)/np.timedelta64(1, 'm')

    if (time_diffs == 0).any():
        print(f"Subject {df['id'].iloc[0]} has repeated glucose measurements. Only the last repeated value is used.")
        _, unique_indx = np.unique(tr, return_index=True)
        tr = tr[sorted(unique_indx)]
        g = g[sorted(unique_indx)]
        time_diffs = np.diff(tr)/np.timedelta64(1, 'm')

    # Calculate dt0 if it is not given
    if dt0 is None:
        dt0 = round(np.nanmedian(time_diffs))
    if dt0 > inter_gap:
        raise ValueError(f"Identified measurements, {dt0} > {inter_gap} minutes apart.")

    # Recalculate dt0 if needed
    if 1440 % dt0 !=0:
        if dt0 > 20:
            dt0 = 20
        else:
            remainder = dt0 % 5
            dt0 += (5-remainder) if remainder > 2 else -remainder

    # Create grid for matrix
    ndays = int(np.ceil((tr[-1]-tr[0]).total_seconds()/86400))+1
    dti = pd.to_timedelta(np.arange(0, ndays*24*60, dt0), unit='m')
    # dti_cum = np.cumsum(dti)
    mind = tr.min().replace(hour=0, minute=0, second=0)
    time_out = mind + dti
    print(f"The time_out is {time_out}")

    #Interpolate
    df_interp = pd.DataFrame({'time': tr, 'glucose': g}).dropna()
    interp = np.interp(
        x = pd.to_numeric(time_out),
        xp = pd.to_numeric(df_interp['time']),
        fp = pd.to_numeric(df_interp['glucose'])
    )
    new = pd.DataFrame({'time': time_out, 'glucose': interp})

    # Handle large time gaps (can't interpolate over large time gaps)
    gap_start = np.where(time_diffs > inter_gap)[0]
    ngaps = len(gap_start)

    if ngaps > 0:
        for idx in range(ngaps):
            time_covered = (new['time'] > tr[gap_start[idx]]) & (new['time'] < tr[gap_start[idx] + 1])
            new.loc[time_covered, 'glucose'] = np.nan


    # Reshape data to days
    gd2d = interp.reshape((ndays,-1))
    actual_dates = pd.date_range(start=mind.normalize(), periods=ndays, freq='D')

    return [gd2d, actual_dates, dt0]