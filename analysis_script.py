from argparse import ArgumentParser
import os.path
import re

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

parser = ArgumentParser(description="Analyze the results of text attacking on news")
parser.add_argument("-i", dest="filename", required=True,
                    help="input file", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))     

# Read input file
args = parser.parse_args()
# total number of data point in the input file
total_count = 0
# the success cases
from_0_count = 0
from_0_skipped = 0
from_0_failed = 0
from_0_to_1_count = 0

from_1_count = 0
from_1_skipped = 0
from_1_failed = 0
from_1_to_0_count = 0

num_replaced_words = []

# [91m0 (55%)[0m --> [92m1 (100%)[0m
case_pattern = r"\[9(.*?)m(.*?) --> \[(.*)" # \((.*?)\)[0m
skipped_pattern = r"(.*)-->(.*?)\[SKIPPED\]\[0m" #r" \[9(.*?)m(.*?) --> \[37m\[SKIPPED\]\[0m"
failed_pattern = r"(.*)-->(.*?)\[FAILED\]\[0m" #r" \[9(.*?)m(.*?) --> \[37m\[FAILED\]\[0m"
from_0_to_1_pattern = "(.*)91m0(.*)92m1(.*)"
from_1_to_0_pattern = "(.*)92m1(.*)91m0(.*)"
attacked_before_pattern = r"\[91m(.*?)\[0m"#"\[91m(.*).+?\[0m"
attacked_after_pattern = r"\[92m(.*?)\[0m"

input_file = args.filename
output_file = open("output.txt", "w")

line = input_file.readline()
while line:
    if re.search(case_pattern, line) is None:
        line = input_file.readline()
        continue
    total_count += 1
    print(line)
    if line.startswith('[91m0'):# re.search(r"\[91m0(.*)", line):
        from_0_count += 1
        if re.search(skipped_pattern, line):
            from_0_skipped += 1
            line = input_file.readline()
            continue
        elif re.search(failed_pattern, line):
            from_0_failed +=1
            line = input_file.readline()
            continue
        match = re.search(from_0_to_1_pattern, line)
        if match is None:
            line = input_file.readline()
            continue
        from_0_to_1_count += 1
        total_original_words = []
        original_nextline = input_file.readline().rstrip()
        while not original_nextline:
            original_nextline = input_file.readline().rstrip()
        while original_nextline:
            total_original_words.extend(re.findall(attacked_before_pattern, original_nextline))
            original_nextline = input_file.readline().rstrip()
        # skip all the empty lines, and get to the first line of attacked article
        attacked_nextline = input_file.readline().rstrip()
        while not attacked_nextline:
            attacked_nextline = input_file.readline().rstrip()
        attacked = ""
        attacked_words_count = 0
        while attacked_nextline and re.search(case_pattern, attacked_nextline) is False: #attacked_nextline.startswith('[91m0') is False:
            attacked_words = re.findall(attacked_after_pattern, attacked_nextline)
            if not attacked_words:
                attacked += "\n" + attacked_nextline
            else:
                num = len(attacked_words)
                for i in range(num):
                    # print(attacked_words[i])
                    replaced_str = "*" + attacked_words[i] + "*" + " (" + total_original_words[attacked_words_count + i] + ")"
                    attacked_nextline = re.sub(attacked_after_pattern, replaced_str, attacked_nextline, 1)
                attacked_words_count += num
                attacked += "\n" + attacked_nextline
            attacked_nextline = input_file.readline().rstrip()
        # print("original")
        # print(total_original_words)
        # print("attacked")
        # print(attacked)
        output_file.write(attacked + "\n")
        num_replaced_words.append(len(total_original_words))
        if re.search(case_pattern, attacked_nextline): 
            line = attacked_nextline
        else:
            # read the next line
            line = input_file.readline()
    elif line.startswith('[92m1'): # re.search(r"\[92m1(.*)", line):
        # print(line)
        from_1_count += 1
        if re.search(skipped_pattern, line):
            from_1_skipped += 1
            line = input_file.readline()
            continue
        elif re.search(failed_pattern, line):
            from_1_failed +=1
            line = input_file.readline()
            continue
        match = re.search(from_1_to_0_pattern, line)
        if match is None:
            line = input_file.readline()
            continue
        from_1_to_0_count += 1
        total_original_words = []
        original_nextline = input_file.readline().rstrip()
        while not original_nextline:
            original_nextline = input_file.readline().rstrip()
        while original_nextline:
            total_original_words.extend(re.findall(attacked_before_pattern, original_nextline))
            original_nextline = input_file.readline().rstrip()
        # skip all the empty lines, and get to the first line of attacked article
        attacked_nextline = input_file.readline().rstrip()
        while not attacked_nextline:
            attacked_nextline = input_file.readline().rstrip()
        attacked = ""
        attacked_words_count = 0
        while attacked_nextline and re.search(case_pattern, attacked_nextline) is False: #attacked_nextline.startswith('[91m0') is False:
            attacked_words = re.findall(attacked_after_pattern, attacked_nextline)
            if not attacked_words:
                attacked += "\n" + attacked_nextline
            else:
                num = len(attacked_words)
                for i in range(num):
                    # print(attacked_words[i])
                    replaced_str = "*" + attacked_words[i] + "*" + " (" + total_original_words[attacked_words_count + i] + ")"
                    attacked_nextline = re.sub(attacked_after_pattern, replaced_str, attacked_nextline, 1)
                attacked_words_count += num
                attacked += "\n" + attacked_nextline
            attacked_nextline = input_file.readline().rstrip()
        # print("original")
        # print(total_original_words)
        # print("attacked")
        # print(attacked)
        output_file.write(attacked + "\n")
        num_replaced_words.append(len(total_original_words))
        if re.search(case_pattern, attacked_nextline): 
            line = attacked_nextline
        else:
            # read the next line
            line = input_file.readline()

    # match = re.search(from_0_to_1_pattern, line)
    # if match is None:
    #     line = input_file.readline()
    #     continue

    # from_0_to_1_count += 1
    # # read a space line
    # input_file.readline()
    # total_original_words = []
    # original_nextline = input_file.readline().rstrip()
    # while original_nextline:
    #     total_original_words.extend(re.findall(attacked_before_pattern, original_nextline))
    #     original_nextline = input_file.readline().rstrip()
    # # skip all the empty lines, and get to the first line of attacked article
    # attacked_nextline = input_file.readline().rstrip()
    # while not attacked_nextline:
    #     attacked_nextline = input_file.readline().rstrip()
    # attacked = ""
    # attacked_words_count = 0
    # while attacked_nextline and re.search(case_pattern, attacked_nextline): #attacked_nextline.startswith('[91m0') is False:
    #     attacked_words = re.findall(attacked_after_pattern, attacked_nextline)
    #     if not attacked_words:
    #         attacked += "\n" + attacked_nextline
    #     else:
    #         num = len(attacked_words)
    #         for i in range(num):
    #             # print(attacked_words[i])
    #             replaced_str = "*" + attacked_words[i] + "*" + " (" + total_original_words[attacked_words_count + i] + ")"
    #             attacked_nextline = re.sub(attacked_after_pattern, replaced_str, attacked_nextline, 1)
    #         attacked_words_count += num
    #         attacked += "\n" + attacked_nextline
    #     attacked_nextline = input_file.readline().rstrip()
    # # print("original")
    # # print(total_original_words)
    # # print("attacked")
    # # print(attacked)
    # output_file.write(attacked + "\n")
    # num_replaced_words.append(len(total_original_words))
    # if re.search(case_pattern, attacked_nextline): 
    #     line = attacked_nextline
    # else:
    #     # read the next line
    #     line = input_file.readline()
    

# for line in lines:
#     if line.startswith('[91m0') is False:
#         continue
#     total_count += 1
#     match = re.search(succuss_case, line)
#     if match is None:
#         continue
#     success_count += 1
        
print("Total count: %d" % (total_count))
print("Originally detected as 'machine' (0): %d" % from_0_count)
print("- success count: %d, success rate: %.2f" % (from_0_to_1_count, from_0_to_1_count/from_0_count))
print("- skipped count: %d" % (from_0_skipped))
print("- fail count: %d" % (from_0_failed))

print("Originally detected as 'human' (1): %d" % from_1_count)
print("- success count: %d, success rate: %.2f" % (from_1_to_0_count, from_1_to_0_count/from_1_count))
print("- skipped count: %d" % (from_1_skipped))
print("- fail count: %d" % (from_1_failed))

print("Average replaced words: %.2f" % (sum(num_replaced_words)/len(num_replaced_words)))


input_file.close()
output_file.close()


