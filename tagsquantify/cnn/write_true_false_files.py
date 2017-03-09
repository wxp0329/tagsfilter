# coding:utf-8
import threading
import threadpool, time,os
from multiprocessing import cpu_count

#输出规定jaccard系数所对应的pairs
def chek_jaccard():
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_key_list.dat') as fr:
        names = fr.readlines()
    with open('/home/wangxiaopeng/0.1_dem.txt') as fr:
        indexes_pair = fr.readlines()
    import shutil
    shutil.rmtree('/home/wangxiaopeng/find_pic/')
    os.mkdir('/home/wangxiaopeng/find_pic/')
    i = 1
    for pair in indexes_pair:
        ab = pair.strip().split(' ')
        pic1 = names[int(ab[0])].strip()
        pic2 = names[int(ab[1])].strip()
        # os.mkdir('/home/wangxiaopeng/find_pic/'+pic1+pic2)
        with open('/media/wangxiaopeng/maxdisk/NUS_dataset/images_210841/' + pic1 + '.jpg') as fr1:
            with open('/home/wangxiaopeng/find_pic/' + str(i) + '_' + pic1 + '.jpg', 'w') as fw1:
                fw1.writelines(fr1.readlines())
        with open('/media/wangxiaopeng/maxdisk/NUS_dataset/images_210841/' + pic2 + '.jpg') as fr2:
            with open('/home/wangxiaopeng/find_pic/' + str(i) + '_' + pic2 + '.jpg', 'w') as fw2:
                fw2.writelines(fr2.readlines())

#保存所有pairs所对应的jaccard系数
def save_jaccard_set():
    with open('/home/wangxiaopeng/intersect_replaced_220341_tags.dat') as fr:
        a = fr.readlines()
    sorted_file = sorted(a, key=lambda ab: float(str(ab).strip().split(' ')[-1]), reverse=True)
    with open('/home/wangxiaopeng/operate/sorted_intersect_replaced_220341_tags.dat', 'w') as fw:
        fw.writelines(a)
    print 'save sorted_intersect_replaced_220341_tags.dat over!!!!'
    # jfw = open('/home/wangxiaopeng/operate/jaccard_set.txt', 'w')
    print 'start add set.............'
    myset = set()
    for i in sorted_file:
        myset.add(i.strip().split(' ')[-1])
    print 'start write set file.......'
    with open('/home/wangxiaopeng/operate/jaccard_set.txt', 'w') as jfw:
        for i in myset:
            jfw.write(i + '\n')
            jfw.flush()
    print 'save  over!!!'


print 'reading file sorted_intersect_replaced_220341_tags.dat...........'
with open('/home/wangxiaopeng/operate/sorted_intersect_replaced_220341_tags.dat') as fr:
    sorted_pairs = fr.readlines()

print 'reading file sorted_intersect_replaced_220341_tags.dat over!!!!!!!!!!'

#分别把正例文件和负例文件写入对应文件夹
def write_files(i, j):
    print 'start write file:', str(i), '...........'
    true_fw = open('/home/wangxiaopeng/operate/true_files/' + str(i) + '_true_file.txt','w')
    false_fw = open('/home/wangxiaopeng/operate/false_files/' + str(i) + '_false_file.txt','w')
    for pair in sorted_pairs[i:j]:
        pair = pair.strip().split(' ')
        if float(pair[-1]) >= 0.4:
            true_fw.write(pair[0] + ' ' + pair[1] + '\n')
            true_fw.flush()
        elif float(pair[-1]) < 0.2:
            false_fw.write(pair[0] + ' ' + pair[1] + '\n')
            false_fw.flush()
    true_fw.close()
    false_fw.close()
    print 'write file:', str(i), 'over !!!!!!!!!!!'


def run_write_true_false_files():
    print 'cpu_count :', cpu_count()
    len_ = len(sorted_pairs)
    i_list = []
    for i in xrange(0, int(1e12), 3000000):  # 3000000 行大约50M数据

        if i + 3000000 >= len_:
            i_list.append([i, len_ + 1])
            break
        i_list.append([i, i + 3000000])
    n_list = [None for i in range(len(i_list))]
    pool = threadpool.ThreadPool(cpu_count())
    requests = threadpool.makeRequests(write_files, zip(i_list, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print 'all of processes excute over !!!!!'


if __name__ == '__main__':
    # run_write_true_false_files()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_key_list.dat') as fr:
        names = fr.readlines()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/images_210841/1348680820.jpg') as fr1:
        with open('/home/wangxiaopeng/find_pic/17558.jpg', 'w') as fw1:
            fw1.writelines(fr1.readlines())
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/images_210841/34802460.jpg') as fr2:
        with open('/home/wangxiaopeng/find_pic/6732.jpg', 'w') as fw2:
            fw2.writelines(fr2.readlines())