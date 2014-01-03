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

__version__ = "2.0.0-SNAPSHOT"

# format:
# line 1 - number of units (nunits)
# line 2 - total number of repeats per unit (nreps)
# line 3:(3+nunits) - starting lines for each unit, i.e. pointers
# to locations in the file where unit data is. Advance to that line
# and scan in nreps lines, which give the number of events per repeat.


def read(fp):
    """Parses fp (a file object) as a toe_lis file and returns a tuple of ragged
    arrays, one for each element in the file.

    """
    from numpy import fromiter
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
        events = [fromiter(lines, 'd', n) for n in n_events]
        out.append(events)
        pos += sum(n_events) + n_repeats

    return tuple(out)


def write(fp, *data):
    """Writes time of event data to fp (a file object) in toe_lis format.

    The data arguments must each be a ragged array containing event times. The
    data can be in any format, as long as it can be iterated (and support
    __len__) at two levels, and the returned values are numeric. Multiple
    objects can be supplied on the command line, each of which is treated as a
    different 'unit' in the toe_lis file; however, each object must have the
    same number of trials.

    """
    from itertools import chain
    output = []
    header = []
    ntrials = None

    header.append(len(data))  # number of units

    ptr = 3 + len(data)       # first data entry
    for unit in data:
        if ntrials is None:
            ntrials = len(unit)
            header.append(ntrials)
        elif ntrials != len(unit):
            raise ValueError("Each unit must have the same number of repeats")
        header.append(ptr + len(output))
        output.extend(len(trial) for trial in unit)
        for trial in unit:
            output.extend(trial)

    for val in chain(header,output):
        fp.write("%r\n" % val)


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
    from itertools import izip
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


class toelis(list):
    """
    A toelis object represents a collection of events. Each event is simply a
    scalar time offset. The events are organized hierarchically into 'trials';
    for example, there may be events occurring in multiple independent channels,
    or the events may be occur in response to multiple presentations of the same
    stimulus.

    This class derives from <list>, and the item access methods have been
    overridden, where appropriate, to return new toelis objects, and convert
    added data to the correct format. Each element of the list contains a 1D
    numpy ndarray.

    toelis():         initialize an empty object
    toelis(iterable): create a new object from a list of event times

    nevents:          total count of events
    range:            min and max event across all trials
    subrange():       return a new toelis with events restricted to a window
    merge():          copy events from one object to this one
    rasterize():      convert ragged array to x,y indices
    """

    def __init__(self, trials=None):
        """
        Constructs the toelis object.

        toelis():         construct an empty object
        toelis(iterable):

        Intialize the object with data from iterable. Each element in iterable
        must be a list of numeric values; these are converted to a numpy
        ndarray. If the data are already in ndarrays, the copy is shallow and
        any manipulations will affect the underlying data; the calling function
        should explicitly copy the data to avoid this.
        """
        if trials is None:
            list.__init__(self)
        else:
            list.__init__(self, (self._convert_data(x) for x in trials))

    @staticmethod
    def _convert_data(trial):
        from numpy import array
        d = array(trial, ndmin=1)
        if d.ndim > 1:
            raise ValueError("Input data must be 1-D")
        return d

    def __getslice__(self, *args):
        return self.__class__(list.__getslice__(self, *args))

    def __setslice__(self, start, stop, trials):
        try:
            list.__setslice__(self, start, stop,
                              (self._convert_data(x, False) for x in trials))
        except TypeError:
            raise TypeError("can only assign an iterable")

    def __setitem__(self, index, trial):
        list.__setitem__(self, index, self._convert_data(trial, False))

    def append(self, trial):
        """Append new trial to end"""
        list.append(self, self._convert_data(trial))

    def extend(self, trials):
        """Add each item in trials to the end of the toelis """
        list.extend(self, (self._convert_data(x) for x in trials))

    def insert(self, index, trial):
        """Insert a new trial before index"""
        list.insert(self, index, self._convert_data(trial))

    def __add__(self, trials):
        """ Add the trials in another object to this one. Shallow copy. """
        from itertools import chain
        return toelis(chain(self, trials))

    def __repr__(self):
        if len(self) < 100:
            return "<%s %d trials, %d events>" % (self.__class__.__name__, len(self), self.nevents)
        else:
            return "<%s %d trials>" % (self.__class__.__name__,len(self))

    def __str__(self):
        return "[" + "\n ".join(tuple(trial).__repr__() for trial in self) + "]"

    def offset(self, offset):
        """ Adds a fixed offset to all the time values in the object.  """
        from numpy import isscalar
        if not isscalar(offset):
            raise TypeError("can only add scalars to toelis events")
        for trial in self:
            trial += offset

    @property
    def nevents(self):
        """ The total number of events in the object """

    @property
    def range(self):
        """ The range of event times in the object """
        if self.nevents==0: return None,None
        mn,mx = zip(*((x.min(),x.max()) for x in self if x.size))
        return (min(mn) or len(mn) and None,
                max(mx) or len(mx) and None)

    def subrange(self, onset=None, offset=None, adjust=False):
        """
        Returns a new toelis object only containing events between onset and
        offset (inclusive). Default values are -Inf and +Inf.

        If <adjust> is True, set times relative to onset
        If <adjust> is a scalar, set times relative to <adjust>
        Default is to leave the times alone
        """
        mintime,maxtime = self.range
        if onset==None: onset = mintime
        if offset==None: offset = maxtime
        if adjust==True:
            adjust = onset
        elif adjust==False:
            adjust = 0
        return toelis([x[((x>=onset) & (x<=offset))] - adjust for x in self])

    def merge(self, newlis, offset=0.0):
        """
        Merge two toelis objects by concatenating events in corresponding
        repeats. For example, if tl1[0]= [1,2,3] and tl2[0]= [4,5,6], after
        tl1.merge(tl2), tl1[0] = [1,2,3,4,5,6]. The events are NOT sorted.

        <offset> is added to all events in newlis
        """
        from numpy import concatenate
        if not len(self)==len(newlis):
            raise ValueError("Number of trials must match")
        for i,trial in enumerate(self):
            self[i] = concatenate([trial, newlis[i] + offset])

    def rasterize(self):
        """
        Rasterizes the data as a collection of x,y points, with the x position
        determined by the event time and the y position determined by the trial
        index. Returns a tuple of arrays, (x,y)
        """
        from numpy import concatenate, ones
        y = concatenate([ones(unit.size, dtype='i') * i for i, unit in enumerate(self)])
        x = concatenate(self)
        return (x, y)

# Variables:
# End:
