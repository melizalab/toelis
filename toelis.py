# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Read and write event-time data in "toelis" format.

Classes
====================
toefile:   read and write data in toe_lis format
toelis:    class for representing time of event data

File format originally developed by Amish Dave.
Copyright (C) Dan Meliza, 2006-2013 (dmeliza@uchicago.edu)
Licensed for use under GNU Public License v2.0
"""
import numpy as nx
import os

__version__ = "1.0.0"

class toefile(object):
    """
    Access toe_lis files. These are flat ASCII-encoded files that can store toe
    data for multiple units and multiple epochs. The files are typically small
    enough that they can be read into memory immediately, so the class only
    defines two methods:

    read():        read toe data
    write(data):   write data to toe_lis file
    """

    format_doc = """
        # line 1 - number of units (nunits)
        # line 2 - total number of repeats per unit (nreps)
        # line 3:(3+nunits) - starting lines for each unit, i.e. pointers
        # to locations in the file where unit data is. Advance to that line
        # and scan in nreps lines, which give the number of events per repeat.
        """

    def __init__(self, filename, *args, **kwargs):
        self.filename = filename
        self.timestamp = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return exc_val

    def read(self):
        """
        Read a toe_lis file. Returns a tuple of toelis objects, one for every
        unit in the file. A brief documentation of the file format can be found
        in the format_doc property.
        """
        if isinstance(self.filename, basestring):
            fp = open(self.filename, 'rU')
        else:
            fp = self.filename
        n_units = int(float(fp.readline()))
        n_repeats = int(float(fp.readline()))
        out = []

        # use this information to check for consistency
        p_units = [int(float(fp.readline())) for i in range(n_units)]
        pos = 2 + n_units + 1

        for unit in range(n_units):
            if pos != p_units[unit]:
                raise IOError, "Corrupted header in %s: unit %d should start on %d" % (fp,unit,p_units[unit])
            n_events = [int(float(fp.readline())) for i in range(n_repeats)]
            events = toelis([float(fp.readline()) for j in range(n)] for n in n_events)
            out.append(events)
            pos += sum(n_events) + n_repeats

        fp.close()
        return tuple(out)

    def write(self, *data):
        """
        Writes time of event data to a toe_lis file. The basic data type is a
        ragged array, each element of which contains a list of event times. The
        data can be in any format, as long as it can be iterated (and support
        __len__) at two levels, and the returned values are numeric. Multiple
        objects can be supplied on the command line, each of which is treated as
        a different 'unit' in the toe_lis file; however, each object must have
        the same number of trials.
        """
        from itertools import chain
        output = []
        header = []
        ntrials = [len(unit) for unit in data]
        for nt in ntrials:
            if not nt==ntrials[0]: raise ValueError, "Each object must have the same length for toe_lis file"
        header.append(len(data))  # number of units
        header.append(ntrials[0]) # number of trials

        ptr = 3 + len(data)       # first data entry
        for unit in data:
            header.append(ptr + len(output))
            output.extend(len(trial) for trial in unit)
            for trial in unit: output.extend(trial)

        if isinstance(self.filename,basestring):
            fp = open(self.filename,'wt')
        else:
            fp = self.filename
        try:
            for val in chain(header,output):
                fp.write("%r\n" % val)
        finally:
            if isinstance(self.filename,basestring): fp.close()

    def close(self):
        """ Set the timestamp of the file (the handle is only opened transiently) """
        if self.timestamp is not None:
            os.utime(self.filename, (self.timestamp, self.timestamp))


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
        d = nx.array(trial, ndmin=1)
        if d.ndim > 1: raise ValueError, "Input data must be 1-D"
        return d

    def __getslice__(self, *args):
        return self.__class__(list.__getslice__(self, *args))

    def __setslice__(self, start, stop, trials):
        try:
            list.__setslice__(self, start, stop,
                              (self._convert_data(x, False) for x in trials))
        except TypeError:
            raise TypeError, "can only assign an iterable"

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

    def sort(self, indices):
        raise NotImplementedError, "Sorting toelis objects not supported"

    def __repr__(self):
        if len(self) < 100:
            return "<%s %d trials, %d events>" % (self.__class__.__name__,len(self),self.nevents)
        else:
            return "<%s %d trials>" % (self.__class__.__name__,len(self))

    def __str__(self):
        return "[" + "\n ".join(tuple(trial).__repr__() for trial in self) + "]"

    def offset(self, offset):
        """ Adds a fixed offset to all the time values in the object.  """
        if not nx.isscalar(offset):
            raise TypeError, " can only add scalars to toelis events"
        for trial in self:
            trial += offset

    @property
    def nevents(self):
        """ The total number of events in the object """
        return sum(x.size for x in self)

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
        if not len(self)==len(newlis):
            raise ValueError, "Number of trials must match"
        for i,trial in enumerate(self):
            self[i] = nx.concatenate([trial, newlis[i] + offset])

    def rasterize(self):
        """
        Rasterizes the data as a collection of x,y points, with the x position
        determined by the event time and the y position determined by the trial
        index. Returns a tuple of arrays, (x,y)
        """
        y = nx.concatenate([nx.ones(unit.size,dtype='i') * i for i,unit in enumerate(self)])
        x = nx.concatenate(self)
        return x,y

# Variables:
# End:
