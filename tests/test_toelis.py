# -*- coding: utf-8 -*-
# -*- mode: python -*-
import toelis
import StringIO

toe1 = """\
1
10
4
4
3
7
14
6
9
18
12
6
7
-590.849975586
-261.550048828
589.100097656
10406.75
-1809.0
452.600097656
5721.25
-1298.29998779
-1173.70001221
-453.0
5747.89990234
5959.54980469
7364.45019531
12782.9501953
-1813.94999695
-726.300048828
-692.650024414
-532.099975586
61.6000976562
532.949951172
5678.35009766
5755.64990234
5999.70019531
6104.64990234
6914.20019531
7251.95019531
10673.2998047
12321.0498047
-1624.25
-858.699951172
578.149902344
5729.85009766
10030.1503906
10382.9003906
-1445.09997559
-1433.29998779
-1424.79998779
-1120.29998779
-821.050048828
5708.54980469
10022.5996094
10074.0996094
10333.75
-66.3000488281
22.4000244141
49.4499511719
458.800048828
605.300048828
794.850097656
5694.60009766
5725.95019531
6012.10009766
6547.75
6895.04980469
7149.5
7188.70019531
9657.09960938
10440.5
10494.5
11831.0
12404.3496094
-541.900024414
-506.400024414
-491.599975586
-174.900024414
-10.5
540.75
676.550048828
5750.0
6018.60009766
10027.8496094
10172.3496094
10390.9003906
-1105.40002441
6035.5
7185.45019531
10166.0
11650.9003906
11693.7998047
-1400.65002441
5715.10009766
5931.89990234
6065.39990234
6789.40039062
10013.1503906
10755.5
"""

toe2 = """\
1
10
4
25
4
16
10
6
3
11
12
2
9
388.899902344
524.649902344
647.300048828
804.850097656
1358.85009766
2058.94995117
2190.54980469
2317.70019531
2905.70019531
3155.10009766
3782.75
3964.70019531
4132.75
4378.04980469
4581.75
4999.20019531
5269.14990234
5344.45019531
5361.89990234
5375.60009766
5750.95019531
6033.10009766
8140.09960938
9136.09960938
9216.84960938
528.75
5145.14990234
5287.25
5297.60009766
516.75
590.699951172
607.649902344
1418.44995117
2328.85009766
5271.20019531
5383.60009766
5397.35009766
5696.35009766
6011.29980469
6101.54980469
6681.45019531
6840.84960938
8248.70019531
9120.79980469
9170.25
-231.300048828
589.949951172
811.75
4703.29980469
4845.14990234
5011.60009766
5322.14990234
5428.0
6716.54980469
8193.5
538.5
605.199951172
666.050048828
2268.20019531
2287.14990234
5136.0
606.649902344
4739.25
5365.70019531
-977.150024414
-908.199951172
-233.849975586
651.649902344
2203.29980469
4845.95019531
4935.5
5249.29980469
5356.20019531
6848.65039062
9176.29980469
-968.150024414
-947.900024414
-922.0
584.5
5008.5
5318.85009766
6773.90039062
6817.95019531
6834.84960938
6843.40039062
8917.45019531
9121.15039062
5336.70019531
6789.84960938
-173.75
13.25
548.25
1419.89990234
5097.85009766
5293.89990234
5402.79980469
9054.59960938
11242.2001953
"""

def compare_arrays(a,b,msg):
    from numpy import all
    a,b = a.squeeze(),b.squeeze()
    assert a.shape==b.shape, "%s: shape %s, expected %s" % (msg, a.shape, b.shape)
    assert all(a==b), "%s: arrays unequal" % msg

def compare_values(a,b,msg):
    from numpy import asarray
    from nose.tools import assert_almost_equal
    if isinstance(b,(float)):
        assert_almost_equal(a,b,places=5,msg="%s = %.6f, expected %.6f" % (msg, a, b))
    elif hasattr(a,'__iter__') or hasattr(b,'__iter__'):
        compare_arrays(asarray(a),asarray(b),msg)
    else:
        assert a == b, "%s = %s, expected %s" % (msg, a, b)

def compare_toelis(tl1,tl2,f):
    compare_values(len(tl2), len(tl1),"Number of units in %s" % f)
    for i,u1 in enumerate(tl1):
        u2 = tl2[i]
        compare_values(len(u2),len(u1),"Number of repeats in %s/%d" % (f,i))
        compare_values(u2.nevents,u1.nevents,"Number of events in %s/%d" % (f,i))
        compare_values(u2.range[0],u1.range[0],"Minimum time in %s/%d" % (f,i))
        compare_values(u2.range[1],u1.range[1],"Maximum time in %s/%d" % (f,i))
        for j,trial in enumerate(u1):
            compare_arrays(trial,u2[j],"Event data in %s/%d/%d" % (f,i,j))

def test01_read():
    global data1, data2
    fp = StringIO.StringIO(toe1)
    data1 = toelis.toefile(fp).read()
    compare_values(len(data1), 1, "Number of units in toe file")
    compare_values(repr(data1[0]), '<toelis 10 trials, 86 events>', "Number of repeats/events in the toe file")

    fp = StringIO.StringIO(toe2)
    data2 = toelis.toefile(fp).read()
    compare_values(len(data2), 1, "Number of units in toe file")
    compare_values(repr(data2[0]), '<toelis 10 trials, 98 events>', "Number of repeats/events in the toe file")

def test02_write():
    # because of precision issues, do a read/write
    global data1
    fp = StringIO.StringIO()
    toelis.toefile(fp).write(*data1)

    fp2 = StringIO.StringIO(fp.getvalue())
    d = toelis.toefile(fp2).read()
    compare_toelis(d,data1,"Copied toelis")

def test03_extend():
    global data1, data2
    data1[0].extend(data2[0])
    compare_values(len(data1), 1, "Number of units in toe file")
    compare_values(repr(data1[0]), '<toelis 20 trials, 184 events>', "Number of repeats/events in the toe file")


# Variables:
# End:
