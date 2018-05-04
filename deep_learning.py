import math
import numpy
import random
import time
import prettytable
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Astr = '0.0976270078546 0.430378732745 0.205526752143 0.0897663659938 -0.152690401322 0.291788226133 -0.124825577475 0.783546001564\t0.927325521002 -0.233116962348 0.583450076165 0.0577898395058 0.136089122188 0.851193276585 -0.857927883604 -0.825741400597\t-0.959563205119 0.665239691096 0.5563135019 0.740024296494 0.957236684466 0.598317128433 -0.0770412754941 0.561058352573\t-0.763451148262 0.279842042655 -0.713293425182 0.889337834099 0.0436966435001 -0.170676120019 -0.470888775791 0.548467378868\t-0.0876993355669 0.136867897737 -0.962420399127 0.235270994152 0.224191445445 0.23386799375 0.887496157029 0.363640598207\t-0.280984198852 -0.125936092401 0.395262391855 -0.879549056741 0.333533430891 0.341275739236 -0.579234877852 -0.74214740469\t-0.369143298152 -0.272578458115 0.140393540836 -0.122796973075 0.976747676118 -0.795910378504 -0.58224648781 -0.67738096423\t0.306216650931 -0.49341679492 -0.0673784542874 -0.511148815997 -0.682060832709 -0.779249717671 0.312659178931 -0.723634097303\t-0.60683527664 -0.262549658678 0.641986459696 -0.805797448414 0.675889814998 -0.807803184212 0.952918930027 -0.0626975967046\t0.953522176381 0.20969103949 0.478527158797 -0.921624415491 -0.434386074847 -0.759606877574 -0.407719604956 -0.762544562092'
#
# Wstr = '0.317983179394 0.414262994515 0.0641474963488 0.69247211937\t0.566601454207 0.265389490939 0.523248053467 0.0939405107584\t0.575946495556 0.929296197576 0.318568952451 0.667410379964\t0.131797862404 0.716327204119 0.289406092947 0.183191362007\t0.58651293481 0.0201075461875 0.828940029217 0.00469547619255\t0.677816536796 0.270007973192 0.735194022123 0.962188545117\t0.24875314352 0.576157334418 0.592041931272 0.572251905791\t0.223081632641 0.952749011517 0.447125378618 0.846408672471'
#
# bstr = '0.699479275318 0.297436950855 0.813797819702 0.396505740847'
#
# dZstr = '0.762206394222 0.162545745272 0.76347072371 0.385063180156\t0.450508559639 0.0026487638534 0.912167269446 0.287980398459\t-0.152289902884 0.212786428256 -0.961613603381 -0.396850366651\t0.320347074985 -0.419844785579 0.236030857998 -0.142462598108\t-0.729051871555 -0.403435348088 0.139929821403 0.181745522496\t0.148650497699 0.306401639714 0.304206540003 -0.137163129132\t0.793093191702 -0.264876259904 -0.128270149469 0.783846710031\t0.612387978092 0.407777167081 -0.799546225375 0.838965227489\t0.428482599098 0.997694013136 -0.701103390684 0.736252114736\t-0.675014130647 0.231119128568 -0.752360034301 0.696016458644'
#
# ZRstr = '1.3565476213 1.45771394424 1.50109965789 1.52996162149\t1.46507229855 0.154860821512 0.815361956687 1.04652279418\t2.26219902 1.79451057522 2.93034410997 1.3123696034\t0.236827022288 0.235670995932 0.818587410666 -0.388271700598\t0.817747243124 0.497036862725 1.68708708207 0.79114179937\t0.767800816841 -1.05707251906 0.453867861946 -0.336912607238\t0.229776388413 -1.06142232183 0.233607024277 -1.48127943772\t-0.600774727804 -0.868839796371 -0.870867878022 -0.762964836374\t0.693232976094 0.280523916818 1.11132084879 -0.0493649852366\t0.234454074768 -0.642654040537 -0.630474086786 -0.384635229845'
#
# dAstr = '0.61463791745 0.138201477229 -0.185633405548 -0.86166600909\t0.394857546289 -0.0929146346439 0.444111198941 0.732764651857\t0.951043010006 0.711606684785 -0.97657183163 -0.280043871043\t0.459981124848 -0.656740645477 0.0420732124083 -0.891324023321\t-0.600006950207 -0.962956411079 0.587395406715 -0.552150623879\t-0.309296638606 0.856162586931 0.408828803847 -0.936322140937\t-0.670611687004 0.242956803 0.154457177208 -0.524214357251\t0.86842799585 0.227931911932 0.0712656060499 0.179819952709\t0.460244059034 -0.376110009041 -0.203557875568 -0.58031250205\t-0.627613988239 0.888744779968 0.479101590099 -0.0190823827649'
#
# Fstr = '-0.5451707440533535 -0.4912870364592141 -0.8839416793522488\t-0.13116674888375845 -0.37640823601179485 0.392686977630919\t-0.2444963214150382 -0.6407926448807304 -0.9506425432173375\t-0.8655007370735028 0.3587855469971346 -0.09260631088790938\t0.0731584222174444 0.7933425860806842 0.9806778947934087\t-0.5662060312030521 0.3261564062002016 -0.4733552465256987\t-0.9586980010685426 0.5167573076722829 -0.3599656983550643\t-0.23307221165620406 0.17663422710721144 0.6620969104723808\t0.25796368718229745 0.7453013108947906 -0.45291593036872846\t0.5960936678251274 -0.6287281113880956 0.9055833139438891'
#
# ystr = '0 0 0 1 0 2 2 2 0 0'

# A = [[float(j) for j in i.split(' ')] for i in Astr.split('\t')]
# W = [[float(j) for j in i.split(' ')] for i in Wstr.split('\t')]
# b = [float(j) for j in bstr.split(' ')]
# dZ = [[float(j) for j in i.split(' ')] for i in dZstr.split('\t')]
# Z_relu = [[float(j) for j in i.split(' ')] for i in ZRstr.split('\t')]
# dA_relu = [[float(j) for j in i.split(' ')] for i in dAstr.split('\t')]
# F = [[float(j) for j in i.split(' ')] for i in Fstr.split('\t')]
# y = [int(j) for j in ystr.split(' ')]


def affine_forward(A, W, b):
    n = len(A)
    d = len(A[0])
    d_dash = len(W[0])

    cache_A = A
    cache_W = W
    cache_b = b

    Z = numpy.zeros((n, d_dash))
    Z = numpy.dot(A, W)
    Z += b

    return Z, cache_A, cache_W, cache_b


def affine_backward(dZ, A, W, b):
    n = len(A)
    d = len(A[0])
    d_dash = len(W[0])

    dA = numpy.zeros((n, d))
    dW = numpy.zeros((d, d_dash))
    db = []

    W = numpy.array(W)
    W_T = W.transpose()

    A = numpy.array(A)
    A_T = A.transpose()

    dA = numpy.dot(dZ, W_T)
    dW = numpy.dot(A_T, dZ)

    for j in range(0, d_dash):
        s = 0
        for i in range(0, n):
            s += dZ[i][j]
        db.append(s)

    return dA, dW, db


def relu_forward(Z):
    n = len(Z)
    d_dash = len(Z[0])

    cZ = [[0 for x in range(d_dash)] for y in range(n)]
    cZ = Z

    for i in range(0, n):
        for j in range(0, d_dash):
            if Z[i][j] > 0:
                continue
            else:
                Z[i][j] = 0.0

    return Z, cZ


def relu_backward(dA_relu, cZ):
    n = len(cZ)
    d_dash = len(cZ[0])

    dZ_relu = [[0 for x in range(d_dash)] for y in range(n)]

    for i in range(0, n):
        for j in range(0, d_dash):
            if cZ[i][j] > 0:
                dZ_relu[i][j] = dA_relu[i][j]
            else:
                dZ_relu[i][j] = 0.0

    return dZ_relu


def cross_entropy(F, y, n):

    reci = -1.0 / n
    fsum = 0

    rows = len(F)
    cols = len(F[0])

    dF = [[0 for xx in range(cols)] for yy in range(rows)]

    for i in range(0, n):
        fsum += F[i][y[i]]
        x = 0
        for k in range(0, 3):
            x += numpy.exp(F[i][k])
        fsum = fsum - numpy.log(x)

    L = reci * fsum

    for i in range(0, rows):
        for j in range(0, cols):
            if j == y[i]:
                temp = 1
            else:
                temp = 0

            par = numpy.exp(F[i][j])
            arg = 0
            for k in range(0, 3):
                arg += numpy.exp(F[i][k])

            dF[i][j] = (reci) * (temp - par / arg)

    return L, dF

def four_layer_network(X, W_set, b_set, y, test):
    A_set = [[] for x in range(4)]
    Z_set = [[] for x in range(4)]
    dA_set = [[] for x in range(4)]
    dW_set = [[] for x in range(4)]
    db_set = [[] for x in range(4)]
    dZ_set = [[] for x in range(4)]

    acache1 = [[] for x in range(3)]
    acache2 = [[] for x in range(3)]
    acache3 = [[] for x in range(3)]
    acache4 = [[] for x in range(3)]

    rcache = [[] for x in range(3)]

    eta = 0.1

    Z_set[0], acache1[0], acache1[1], acache1[2] = affine_forward(X, W_set[0], b_set[0])
    A_set[0], rcache[0] = relu_forward(Z_set[0])

    Z_set[1], acache2[0], acache2[1], acache2[2] = affine_forward(A_set[0], W_set[1], b_set[1])
    A_set[1], rcache[1] = relu_forward(Z_set[1])

    Z_set[2], acache3[0], acache3[1], acache3[2] = affine_forward(A_set[1], W_set[2], b_set[2])
    A_set[2], rcache[2] = relu_forward(Z_set[2])

    F, acache4[0], acache4[1], acache4[2] = affine_forward(A_set[2], W_set[3], b_set[3])

    classifications = [0 for x in range(len(F))]

    nn = len(F)

    points = 0

    if test == 1:
        for x in range(len(F)):
            classifications[x] = numpy.argmax(F[x])
            if classifications[x] == y[x]:
                points = points + 1
        return classifications, points

    loss, dF = cross_entropy(F, y, nn)

    dA_set[2], dW_set[3], db_set[3] = affine_backward(dF, acache4[0], acache4[1], acache4[2])
    dZ_set[2] = relu_backward(dA_set[2], rcache[2])

    dA_set[1], dW_set[2], db_set[2] = affine_backward(dZ_set[2], acache3[0], acache3[1], acache3[2])
    dZ_set[1] = relu_backward(dA_set[1], rcache[1])

    dA_set[0], dW_set[1], db_set[1] = affine_backward(dZ_set[1], acache2[0], acache2[1], acache2[2])
    dZ_set[0] = relu_backward(dA_set[0], rcache[0])

    dX, dW_set[0], db_set[0] = affine_backward(dZ_set[0], acache1[0], acache1[1], acache1[2])

    for i in range(5):
        for j in range(256):
            W_set[0][i][j] = W_set[0][i][j] - eta * dW_set[0][i][j]

    for i in range(256):
        for j in range(256):
            W_set[1][i][j] = W_set[1][i][j] - eta * dW_set[1][i][j]

    for i in range(256):
        for j in range(256):
            W_set[2][i][j] = W_set[2][i][j] - eta * dW_set[2][i][j]

    for i in range(256):
        for j in range(3):
            W_set[3][i][j] = W_set[3][i][j] - eta * dW_set[3][i][j]

    for i in range(256):
        b_set[0][i] = b_set[0][i] - eta * db_set[0][i]
        b_set[1][i] = b_set[1][i] - eta * db_set[1][i]
        b_set[2][i] = b_set[2][i] - eta * db_set[2][i]

    for i in range(3):
        b_set[3][i] = b_set[3][i] - eta * db_set[3][i]

    return loss, points


def minibatch_GD(epoch, loss_dict, over_acc, res, actions):
    wsp = 0.01

    W1 = [[(random.random() * wsp) for y in range(256)] for x in range(5)]
    W2 = [[(random.random() * wsp) for y in range(256)] for x in range(256)]
    W3 = [[(random.random() * wsp) for y in range(256)] for x in range(256)]
    W4 = [[(random.random() * wsp) for y in range(3)] for x in range(256)]

    W_set = [W1, W2, W3, W4]

    b1 = [0 for x in range(256)]
    b2 = [0 for x in range(256)]
    b3 = [0 for x in range(256)]
    b4 = [0 for x in range(3)]

    b_set = [b1, b2, b3, b4]

    fd = open("expert_policy.txt", "r")
    data = fd.readlines()
    N = len(data)

    ff = N/128

    for e in range(epoch):
        #print e
        fd = open("expert_policy.txt", "r")
        data = fd.readlines()
        numpy.random.shuffle(data)
        for i in range(N / 128):
            batch_data = []
            loss_arr = []
            X = [[0 for z in range(5)] for x in range(128)]
            y = [0 for x in range(128)]
            for x in range(128):
                line = data[(i * 128) + x]
                batch_data.append(line)
            for a in range(128):
                xxx = [float(jj) for jj in batch_data[a].split(' ')]
                for j in range(0, 4):
                    X[a][j] = xxx[j]
                y[a] = int(xxx[5])

            loss, score = four_layer_network(X, W_set, b_set, y, 0)
            loss_arr.append(loss)

        final_results = {}

        for i in range(N / 128):
            batch_data = []
            X = [[0 for z in range(5)] for x in range(128)]
            y = [0 for x in range(128)]
            for x in range(128):
                line = data[(i * 128) + x]
                batch_data.append(line)
            for a in range(128):
                xxx = [float(jj) for jj in batch_data[a].split(' ')]
                for j in range(0, 4):
                    X[a][j] = xxx[j]
                y[a] = int(xxx[5])

            results, score = four_layer_network(X, W_set, b_set, y, 1)
            accuracy = (score / 128.0) * 100.0
            final_results[i] = accuracy

        lolo = 0

        for i in range(0, len(final_results)):
            lolo = lolo + final_results[i]

        overall_accuracy = lolo / ff

        avg_loss = (numpy.sum(loss_arr))/ff
        loss_dict[e] = avg_loss
        over_acc[e] = overall_accuracy

        if (e == epoch-1):
            for i in range(N / 128):
                batch_data = []
                X = [[0 for z in range(5)] for x in range(128)]
                y = [0 for x in range(128)]
                for x in range(128):
                    line = data[(i * 128) + x]
                    batch_data.append(line)
                for a in range(128):
                    xxx = [float(jj) for jj in batch_data[a].split(' ')]
                    for j in range(0, 4):
                        X[a][j] = xxx[j]
                    y[a] = int(xxx[5])

                actions[i] = y
                res[i], score = four_layer_network(X, W_set, b_set, y, 1)

    #print "Total accuracy: "+str(over_acc[epoch-1])+" %"
    #print "Misclassification error: "+str(1-(over_acc[epoch - 1])/(100))

    return loss_dict, over_acc, W_set, b_set, res, actions

# Runner code

loss_dict = {}
over_acc = {}
ress = {}
acts = {}

ss = time.time()
ld, oa, ww, bb, rr, aa = minibatch_GD(600, loss_dict, over_acc, ress, acts)

# Saving W and b vectors for testing with part 1

ffdd = open("vectors.txt", "a+")
ffdd.write("W")
ffdd.write('\n')
ffdd.write(str(ww))
ffdd.write('\n')
ffdd.write("b")
ffdd.write('\n')
ffdd.write(str(bb))
ffdd.close()

# Confusion Matrix

conMat = []

for i in range(3):
    column = []

    for j in range(3):
        g_count = 0
        b_count = 0
        x = 0

        while (x < len(rr)):

            for y in range(0, 128):

                testClass = rr[x][y]
                solClass = aa[x][y]

                if i == testClass:
                    g_count = g_count + 1

                if (j == solClass) and (i == testClass):
                    b_count = b_count + 1

            x = x + 1

        if g_count == 0:
            val = 0
        else:
            val = (b_count/float(g_count))*(100.0)

        column.append(val)

    conMat.append(column)

conMat = numpy.array(conMat)
p = prettytable.PrettyTable()

for row in conMat:

    p.add_row(row)

#print p.get_string(header=False, border=False)

# Running Time

ee = time.time()
#print "Elapsed time:", str(ee-ss)+" s"

# Graph Plots

plt.subplot(1, 2, 1)
plt.plot(list(oa.keys()), list(oa.values()))
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy in %")

plt.subplot(1, 2, 2)
plt.plot(list(ld.keys()), list(ld.values()))
plt.xlabel("Number of Epochs")
plt.ylabel("Average Loss")

plt.tight_layout()
plt.savefig("MP4_Graphs.png")



