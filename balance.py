from collections import Counter
from collections import defaultdict
import pandas as pd
def balance_windows(win_df):

    labels_count_dict = Counter(win_df['win_label'])
    print (labels_count_dict)
    # min_numoflabels = min(labels_count_dict.values())
    # max_numoflabels = max(labels_count_dict.values())
    # print "the min number is : "
    # print min_numoflabels
    # print "the max number is: "
    # print max_numoflabels

    L = win_df['win_label'].tolist()
    d = defaultdict(int)
    for i in L:
        d[i] += 1
    label_with_maxOccurrance = max(d.iteritems(), key=lambda x: x[1])
    # label_with_maxOccurrance (label, number of occurance)
    # print label_with_maxOccurrance

    print ("original size of win_df: " + str(len(win_df)))

    for cur_label in set(win_df['win_label'].tolist()):
        if cur_label == label_with_maxOccurrance[0]:
            continue
        cur_label_count = labels_count_dict[cur_label]
        # print "cur_label is: " + str(cur_label)
        # print "cur_label_count is: " + str(cur_label_count)

        copy_times = int(round(float(label_with_maxOccurrance[1])/cur_label_count -1,0))
        # print "copy_times is :" + str(copy_times)

        if copy_times > 0:
            cur_label_df = win_df[(win_df['win_label'] == cur_label)]
            # print "cur_label_df : " + str(len(cur_label_df))
            win_df = win_df.append(pd.concat([cur_label_df]*copy_times, ignore_index = True))
            # print "added " + str(labels_count_dict[cur_label] * copy_times) + " label " + str(cur_label)
            # print "now the size of win_df is: " + str(len(win_df))
        # else:
        #     print "no need to copy"

    print ("after balance: ")
    labels_count_dict = Counter(win_df['win_label'])
    print (labels_count_dict)

    return win_df

