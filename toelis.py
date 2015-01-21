# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Read, write, and process time-of-event data.

The functions in this module work with time-of-event data represented as ragged
arrays, which are sequences of arrays of event times. Each component array in
the ragged array corresponds to a different onset time, experimental condition,
etc.

This module takes a dynamic, functional approach; functions do not modify their
arguments and are fairly generic. Some functions may require the component
arrays to support broadcasting. Functions return generators when possible.

File format originally developed by Amish Dave.
Copyright (C) Dan Meliza, 2006-2013 (dmeliza@uchicago.edu)
Licensed for use under GNU Public License v2.0

"""

__version__ = "2.1.0"

# format:
# line 1 - number of units (nunits)
# line 2 - total number of repeats per unit (nreps)
# line 3:(3+nunits) - starting lines for each unit, i.e. pointers
# to locations in the file where unit data is. Advance to that line
# and scan in nreps lines, which give the number of events per repeat.


def read(fp, units=None):
    """Parses fp (a file object) as a toe_lis file and returns a tuple of ragged
    arrays, one for each element in the file.

    units - if not None (the default), sets the physical units of the event
    times. Most toelis files have units of milliseconds, but this is not set to
    retain backwards compatibility.

    """
    from numpy import fromiter
    from quantities import millisecond
    try:
        from __builtin__ import range
    except ImportError:
        from builtins import range

    out = []
    # all lines are parsed as floats first because some very old toelis
    # files have counts stored as floats
    lines = (float(l) for l in fp)
    n_units = int(next(lines))
    n_repeats = int(next(lines))

    # use this information to check for consistency
    p_units = fromiter(lines, 'i', n_units)
    pos = 2 + n_units + 1

    for unit in range(n_units):
        if pos != p_units[unit]:
            raise IOError("Corrupted header in %s: unit %d should start on %d" %
                          (file, unit, p_units[unit]))
        n_events = fromiter(lines, 'i', n_repeats)
        if units is not None:
            events = [fromiter(lines, 'd', n) * units for n in n_events]
        else:
            events = [fromiter(lines, 'd', n) for n in n_events]

        out.append(events)
        pos += sum(n_events) + n_repeats

    return tuple(out)


def write(fp, *data, **kwargs):
    """Writes time of event data to fp (a file object) in toe_lis format.

    The data arguments must each be a ragged array containing event times. The
    data can be in any format, as long as it can be iterated (and support
    __len__) at two levels, and the returned values are numeric. Multiple
    objects can be supplied on the command line, each of which is treated as a
    different 'unit' in the toe_lis file; however, each object must have the
    same number of trials.

    If data have units, event times are rescaled to milliseconds (the standard
    for toelis files); if data are dimensionless, times are assumed to be
    milliseconds.

    Optional arguments:
    format: set format for event times. By default '%r' (full precision)

    """
    from numpy import asarray
    from itertools import chain
    output = []
    header = []
    ntrials = None
    fmt = kwargs.get("format", "%r") + "\n"

    header.append(len(data))  # number of units

    ptr = 3 + len(data)       # first data entry
    for unit in data:
        if ntrials is None:
            ntrials = len(unit)
            header.append(ntrials)
        elif ntrials != len(unit):
            raise ValueError("Each unit must have the same number of repeats")
        header.append(ptr + len(output))
        output.extend("%d\n" % len(trial) for trial in unit)
        for trial in unit:
            if hasattr(trial, "units"):
                trial = asarray(trial.rescale("ms"))
            output.extend(fmt % e for e in trial)

    for val in header:
        fp.write("%d\n" % val)
    fp.writelines(output)



def count(x):
    """Returns the number of events in a ragged array x"""
    return sum(len(y) for y in x)


def range(x):
    """Returns the minimum and maximum values in a ragged array y. If the number
    of events is zero, returns (None, None)

    """
    try:
        return (min(min(y) for y in x), max(max(y) for y in x))
    except ValueError:
        return (None, None)


def offset(x, val):
    """Returns a lazy copy of x with val subtracted from every value.

    Component arrays must support broadcasting.

    """
    return (y - val for y in x)


def subrange(x, onset=None, offset=None):
    """Returns a lazy copy of x with values between onset and offset (inclusive).

    Component arrays must support broadcasting.

    """
    return (y[(y >= onset) & ~(y > (offset))] for y in x)


def merge(x, y):
    """Returns a new lazy ragged array with events in corresponding
    elements of x and y merged. Returned arrays are not sorted.

    >>> tl1 = [[1,2,3]]
    >>> tl2 = [[4,5,6], [7,8,9]]
    >>> list(merge(tl1, tl2))

    """
    try:
        from itertools import izip
    except ImportError:
        izip = zip
    from numpy import concatenate
    return (concatenate((a, b)) for a, b in izip(x, y))


def rasterize(x):
    """Rasterize the ragged array x as a lazy sequence of (x, y)

    The y values of each tuple are the values in the arrays, and the x values
    are the index of the array.

    """
    for i, y in enumerate(x):
        for v in y:
            yield i, v


# Variables:
# End:
