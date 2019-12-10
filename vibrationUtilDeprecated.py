

def mat2array(data):
    y = data['Time_Point1']['y_values'][0][0][0][0][0][0]

    x_init  = data['Time_Point1']['x_values'][0][0][0][0][0][0][0]
    x_step  = data['Time_Point1']['x_values'][0][0][0][0][1][0][0]
    x_quant = data['Time_Point1']['x_values'][0][0][0][0][2][0][0]

    x = np.arange(x_init, x_step*(x_quant), x_step)
    return (x,y)

def generateScratchFromMat(parser=mat2array):
    for file in files:
        if not ('.mat' in file):    continue
        if file == "ignoreme.m":    continue
        if file == "scratch":       continue
        if 'asv' in file:           continue


        if os.path.isfile(dataSetRawPath + '\\scratch\\' + file[0:-4] +'.txt'):
            pickledFile = open(dataSetRawPath + '\\scratch\\' + file[0:-4] +'.txt', 'rb')
            D = (pickle.load(pickledFile))#.reshape(1, -1) + 80
            try:
                plotD(librosa.feature.rms(librosa.istft(D)))
            except:
                plotD(D)
                feature = librosa.feature.rms
                a = feature(librosa.istft(D)).shape; a = a[1];
                sig = -feature(librosa.istft(D))[0]
                sig = sig - np.median(sig)
                sig = sig/np.std(sig)
                plt.plot(np.arange(1,a+1), sig);
                plt.show()
            continue

        data = scipy.io.loadmat(dataSetRawPath+"\\"+file)

        (x,y) = parser(data)


        #plt.plot(x,y); plt.show()

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y,)), ref=np.max)
        plotD(D)

        if DEBUG: print("Saving    at: " + dataSetRawPath + '\\scratch\\' + file[0:-4] +'.txt')
        pickle.dump(D, open(dataSetRawPath + '\\scratch\\' + file[0:-4] +'.txt', 'wb'))

        if   "impactos1" in file:   pass #batida1in.append(D)
        elif "impactos2" in file:   pass #batida4in.append(D)
        elif "impactos3" in file:   pass #batida8in.append(D)
        else:                       raise Exception("Unknown test")


def legacy(): #used to be out of main
    telegram_bot_sendtext("Hello there.")
    convFilters = 4
    filterSize = 3
    poolingSize = 4
    dropOutProportion = 0.1

    generateScratchFromCsv()
    (trainingSet, avaliatiSet, avaliati) = prepareBatches(distancesDict)

    # convProps = [{'nFilters': 16, 'convSize': (4, 7),   'dropOut': 0.12,    'Pooling':(1,1)},
    #              {'nFilters': 32, 'convSize': (16, 1),  'dropOut': 0.12,    'Pooling':(1,1)},
    #              {'nFilters': 64, 'convSize': (8, 1),   'dropOut': 0.12,    'Pooling':(1,1)},
    #              {'nFilters': 32, 'convSize': (16, 1),  'dropOut': 0.12,    'Pooling':(1,1)},
    #              {'nFilters': 64, 'convSize': (8, 1),   'dropOut': 0.12,    'Pooling':(1,1)}]

    run=1

    minConvLayers   = 2
    MaxConvLayers   = 8
    minNfilters     = 0 # represents powers of 2
    MaxNfilters     = 5 # represents powers of 2
    minConvSizesY   = 1 # represents powers of 2
    MaxConvSizesY   = 5 # represents powers of 2
    minConvSizesX   = 1
    MaxConvSizesX   = 7
    mindropOuts     = 0.01
    MaxdropOuts     = 0.25
    minPoolingsY    = 0 # represents powers of 2
    MaxPoolingsY    = 0 # represents powers of 2
    minPoolingsX    = 1
    MaxPoolingsX    = 1
    MaxRuns         = 4

    possibleLayers = []

    for nf in [2 ** i for i in range(minNfilters, MaxNfilters + 1)]:
        for csy in [2 ** i for i in range(minConvSizesY, MaxConvSizesY + 1)]:
            for csx in [minConvSizesX, MaxConvSizesX]:
                for dp in np.linspace(mindropOuts, MaxdropOuts, 3):
                    for pool in range(2):
                        if pool % 2 == 0:
                            pooling = None
                        else:
                            pooling = (1,1)
                        l = {'nFilters': nf, 'convSize': (csy, csx), 'dropOut': dp, 'Pooling': pooling}
                        if DEBUG: print(l)
                        possibleLayers.append(l)

    print("Possible " + str(len(possibleLayers)) + " layers")

    ammountOfTests = int(4*60*24*4.5/MaxRuns)
    lastMilestone = -1

    for i in range(1):
        if floor(100*i/ammountOfTests) > lastMilestone:
            lastMilestone = floor(100*i/ammountOfTests)
            telegram_bot_sendtext(str(lastMilestone) + "% done...")

        convProps = []
        for j in range((i % (MaxConvLayers - minConvLayers)) + 2):
            convProps.append(possibleLayers[random.randint(0,len(possibleLayers)-1)])

        for run in range(1):
            try:
                for i in convProps:
                    print(i)
                ret = main(convProps, givenBatches=(trainingSet, avaliatiSet), epochs=1, dictOfOutputs=distancesDict, comments="Run" + str(run), modelVerbose=0, save=False)
                print(ret)
                if ret != None:
                    telegram_bot_sendtext('fine:' + ret)

            except Exception as e:
                print('exce:'+ str(e))
                telegram_bot_sendtext('raised')
                telegram_bot_sendtext('Problem:' + str(e))

    telegram_bot_sendtext("100% done!!!")

    #
    convProps = [{'nFilters': 2**3,    'convSize': (1, 1), 'dropOut': 0.12,  'Pooling':  None},
                 {'nFilters': 2**2,    'convSize': (1, 1), 'dropOut': 0.05,  'Pooling':  None},
                 {'nFilters': 2**2,    'convSize': (1, 1), 'dropOut': 0.05,  'Pooling':  None},
                 {'nFilters': 2**1,    'convSize': (1, 1), 'dropOut': 0.05,  'Pooling':  None},
                 {'nFilters': 2**1,    'convSize': (1, 7), 'dropOut': 0.05,  'Pooling':  None}]

    # convProps = [{'nFilters': 8, 'convSize': (8, 1),   'dropOut': 0.01,    'Pooling':(1,1)},
    #              {'nFilters': 16, 'convSize': (8, 7),  'dropOut': 0.13,    'Pooling':(1,1)}]  constante 88%

    #
    # convProps = [{'nFilters': 16, 'convSize': (4, 7),   'dropOut': 0.12,    'Pooling':(1,1)},
    #              {'nFilters': 32, 'convSize': (16, 1),  'dropOut': 0.12,    'Pooling':(1,1)},
    #              {'nFilters': 64, 'convSize': (8, 1),   'dropOut': 0.12,    'Pooling':(1,1)},
    #              {'nFilters': 32, 'convSize': (16, 1),  'dropOut': 0.12,    'Pooling':(1,1)},
    #              {'nFilters': 64, 'convSize': (8, 1),   'dropOut': 0.12,    'Pooling':(1,1)}]
    main(convProps, givenBatches=(trainingSet, avaliatiSet), epochs=1000, dictOfOutputs=distancesDict,
         comments="Run" + str(run), modelVerbose=1, save=True)

    telegram_bot_sendtext("shutting down!")
    #subprocess.call(["shutdown", "-f", "-s", "-t", "60"])