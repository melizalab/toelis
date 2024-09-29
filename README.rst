toelis
------

|ProjectStatus|_ |Version|_ |BuildStatus|_ |License|_ |PythonVersions|_

.. |ProjectStatus| image:: https://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |Version| image:: https://img.shields.io/pypi/v/toelis.svg
.. _Version: https://pypi.python.org/pypi/toelis/

.. |BuildStatus| image:: https://github.com/melizalab/toelis/actions/workflows/tests-python.yml/badge.svg
.. _BuildStatus: https://github.com/melizalab/toelis/actions/workflows/tests-python.yml

.. |License| image:: https://img.shields.io/pypi/l/toelis.svg
.. _License: https://opensource.org/license/bsd-3-clause/

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/toelis.svg
.. _PythonVersions: https://pypi.python.org/pypi/toelis/


**Toelis** stands for *Time Of Event LISt* and refers to a type of data
consisting of a series of events occurring at discrete times. Each event
is associated with a specific *channel* (i.e. source), and *trial*
(i.e. presentation of a stimulus or set of experimental conditions), and
is defined by a positive or negative temporal from a defined reference
time. This project (1) defines a file format for storing toelis data in
plain text files (original developed by Amish Dave), and (2) contains a
Python module with data structures for storing toelis data and functions
for reading and writing files in the toelis format.

toelis files
------------

Toelis files must be ISO-8859-1 or UTF-8 text encoding, and must have
the format specified by the following
`ABNF <https://tools.ietf.org/html/rfc5234>`__ grammar:

.. code:: abnf

   toelis-file   = num-channels NL num-trials NL *channel-index *channel
   num-channels  = 1*DIGIT
   num-reps      = 1*DIGIT
   channel-index = 1*DIGIT NL           ; starting line for each channel
                                        ; number of channel-index items must equal num-channels
   channel       = event-counts events
   event-counts  = *(1*DIGIT NL)        ; number of events in each repetition
                                        ; number of event-count lines must equal num-reps
   events        = *(time NL)
   time          = [%x1D] 1*DIGIT [%x1E *DIGIT] ; floating point number
   NL            = CR / CRLF / LF       ; all file encodings accepted

The ``time`` elements indicate a positive or temporal interval from an
externally defined reference time (e.g., the onset of a stimulus), and
should be in units of milliseconds.

Informally, the file consists of a series of integers and floating point
numbers on separate lines. The first two lines give the number of
channels and trials in the file. For each channel, there is a line
giving the offset of the channel in the file. These offsets indicate the
start of a block which begins with a series of integers giving the
number of events in each trial, followed by a series of floating point
numbers giving the time of the events. All channels must have the same
number of trials.

python module
-------------

The toelis module requires numpy 1.19+ and has been tested on CPython 3.7-3.12
and PyPy 3.7-3.10. To install:

.. code:: bash

   pip install toelis

Or to install from source:

.. code:: bash

   python setup.py install

The module contains functions for reading and writing toelis files, as
well as for manipulating toelis data, which are represented as lists of
numpy arrays. The functions are well-documented within Python.

An example of loading two files, concatenating the trials, restricting
the trials to a time window, and writing a new file. The functional
programming style allows for compact, expressive syntax:

.. code:: python

   import toelis as tl

   files = ('file1.toe_lis', 'file2.toe_lis')
   t_merged = tl.merge(tl.read(open(fname, 'rU'))[0] for fname in files)
   with open('merged.toe_lis', 'wt') as fp:
       tl.write(fp, tl.subrange(t_merge, 0, 1000))

acknowledgments and license
---------------------------

The toelis format was originally developed by Amish Dave
(http://amishdave.net) for Dan Margoliash’s lab at the University of
Chicago (http://margoliashlab.uchicago.edu). The formalized ABNF grammar
and python implementation were written by Dan Meliza
(http://meliza.org).

See COPYING for license information.
