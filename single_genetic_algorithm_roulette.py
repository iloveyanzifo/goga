import datetime
import random

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.utils import shuffle

import single_init_dataset

profile_arr = single_init_dataset.profile_arr
best_score_so_far = None
crossover_rate = 0.1
mutation_rate = 0.2
iteration_num = 3000
round_num = 1


def get_init_population_list():
    result = []
    init_population = 30
    for _ in range(init_population):
        obj = get_fitness_init()
        result.append(obj)
    return result


def scheduling(current_df, _machine_current_date, batch):
    current_capacity = 0
    capacity = 30
    # 找出目前沒做完的
    tar_df = current_df[(current_df['status'] != 'DONE')]
    # 先取出 1 筆
    first_df = tar_df.head(1)
    current_profile = first_df['profile'].values[0]
    tar_df = current_df[(current_df['status'] != 'DONE') & (current_df['profile'] == current_profile)]
    id_list = []
    flag = False
    if len(tar_df) > 0:
        for index, row in tar_df.iterrows():
            sample_size = row['sample_size']
            current_capacity += sample_size
            if current_capacity > capacity:
                continue
            current_df.loc[index, 'status'] = 'WAIT'
            current_df.loc[index, 'batch'] = batch
        id_list = current_df[current_df['status'] == 'WAIT'].index
        if len(id_list) > 0:
            tar_rows = current_df.loc[id_list].sort_values(by=['arrival_date'])
            last_rows = tar_rows.iloc[-1:]
            make_span = int(last_rows['make_span'])
            a_date = last_rows['arrival_date'].values[-1]
            last_arrival_date = datetime.datetime.strptime(a_date, '%Y-%m-%d %H:%M:%S')
            if _machine_current_date is None:
                next_open_date = last_arrival_date + datetime.timedelta(hours=make_span)
                _machine_current_date = next_open_date
                this_start_date = last_arrival_date
            else:
                if last_arrival_date > _machine_current_date:
                    this_start_date = last_arrival_date
                    next_open_date = last_arrival_date + datetime.timedelta(hours=make_span)
                else:
                    this_start_date = _machine_current_date
                    next_open_date = _machine_current_date + datetime.timedelta(hours=make_span)
                _machine_current_date = next_open_date
            current_df.loc[id_list, 'start_date'] = this_start_date
            current_df.loc[id_list, 'end_date'] = next_open_date
            current_df.loc[id_list, 'status'] = 'DONE'
            for index in id_list:
                row = current_df.loc[index]
                weight = row['weight']
                size = row['sample_size']
                if type(row['due_date']) == str:
                    due_date = datetime.datetime.strptime(row['due_date'], '%Y-%m-%d %H:%M:%S')
                else:
                    due_date = row['due_date']
                if type(row['end_date']) == str:
                    end_date = datetime.datetime.strptime(row['end_date'], '%Y-%m-%d %H:%M:%S')
                else:
                    end_date = row['end_date']
                if end_date > due_date:
                    current_df.loc[index, 'result'] = 'FAIL'
                    delay = end_date - due_date
                    current_df.loc[index, 'delay'] = delay
                    current_df.loc[index, 'score'] = round(delay.total_seconds() * weight * size / 60 / 60, 2)
    else:
        flag = True
    return _machine_current_date, flag, list(id_list)


def get_fitness_init():
    current_df_len = len(pd.read_csv('result/init_data.csv'))
    chromosome = []
    batch_id = 1
    current_df_ = pd.read_csv('result/init_data.csv')
    machine_current_date = datetime.datetime(2022, 5, 1, 0, 0, 0)
    # 亂數排列df
    current_df_ = shuffle(current_df_)
    while current_df_len > 0:
        machine_current_date, next_flag, current_index_arr = scheduling(current_df_, machine_current_date, batch_id)
        if next_flag:
            break
        current_df_len = len(current_df_[current_df_['status'] != 'DONE'])
        chromosome.extend(current_index_arr)
        batch_id += 1
    count_fail = len(current_df_[current_df_['result'] == 'FAIL'])
    score = current_df_['score'].sum()
    fr = (round(count_fail / len(current_df_), 4) * 100)
    fs = round(score, 2)
    return {'fr': fr, 'fs': fs, 'chromosome': chromosome}


def crossover(father, mother):
    index = round(len(father) * crossover_rate)
    start_index = random.randint(0, index)
    end_index = start_index + index
    genetic_cut = father[start_index:end_index]
    mother = [item for item in mother if item not in genetic_cut]
    head = mother[0:start_index]
    foot = mother[start_index:len(mother)]
    offspring = head + genetic_cut + foot
    return offspring


def get_offspring_list(data):
    result = []
    total = data.sum()['fs']
    weight = []
    for idx, row in data.iterrows():
        s = round(((total - row['fs']) / total), 4)
        weight.append(s)
    for _ in range(30):
        father_index = random.choices(data.index.tolist(), weights=weight, k=1)
        mother_index = random.choices(data.index.tolist(), weights=weight, k=1)
        father_ = data.iloc[father_index[0]]['chromosome']
        mother_ = data.iloc[mother_index[0]]['chromosome']
        if type(father_) == str:
            father_ = eval(father_)
        if type(mother_) == str:
            mother_ = eval(mother_)
        offspring = crossover(father_, mother_)
        result.append(offspring)
    return result


def get_mutation_list(data):
    result = []
    for offspring in data:
        index = round(len(offspring) / 2)
        start_index = random.randint(0, index)
        end_index = random.randint(index, len(offspring) - 1)
        offspring = swapPositions(offspring, start_index, end_index)
        result.append(offspring)
    return result


def swapPositions(lst, pos1, pos2):
    lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
    return lst


def get_fitness_from_mutation_list(data):
    result = []
    for chromosome in data:
        temp = []
        current_df = pd.read_csv('result/init_data.csv')
        for operate in chromosome:
            temp.append(current_df.iloc[operate])
        current_df_ = pd.DataFrame(temp)
        batch_id = 1
        machine_current_date = None
        current_df_len = len(current_df_)
        _chromosome = []
        while current_df_len > 0:
            machine_current_date, next_flag, current_index_arr = scheduling(current_df_, machine_current_date, batch_id)
            if next_flag:
                break
            current_df_len = len(current_df_[current_df_['status'] != 'DONE'])
            _chromosome.extend(current_index_arr)
            batch_id += 1
        count_fail = len(current_df_[current_df_['result'] == 'FAIL'])
        score = current_df_['score'].sum()
        if score < best_score_so_far['fs']:
            current_df_.to_csv('best_so_far.csv', index=False)
        fr = (round(count_fail / len(current_df_), 4) * 100)
        fs = round(score, 2)
        result.append({'fr': fr, 'fs': fs, 'chromosome': _chromosome})
    return result


def get_select_parent(data):
    global best_score_so_far
    row_arr = []
    for _ in range(30):
        row = data.iloc[i]
        if best_score_so_far is None:
            best_score_so_far = row
        if row['fs'] < best_score_so_far['fs']:
            best_score_so_far = row
        row_arr.append(row)
    result = pd.DataFrame(row_arr)
    return result


def do_algorithm(data, b_s):
    # 取得親代 1 筆
    data = data.sort_values(by=['fs'], ascending=True)
    select_parent = get_select_parent(data)
    # 得到後代 2 筆
    offspring_list = get_offspring_list(select_parent)
    # 得到突變後代 2 筆
    mutation_list = get_mutation_list(offspring_list)
    # 得到新的適應函數
    new_parent = get_fitness_from_mutation_list(mutation_list)
    temp_df = pd.DataFrame(new_parent)
    best_fs = temp_df[temp_df['fs'] == temp_df['fs'].min()].iloc[0]
    new_select_parent = temp_df.sort_values(by=['fs'], ascending=True)
    return new_select_parent, best_fs


def process():
    # step1 生成母體30筆 最後產生母體arr ipl.csv
    init_population_list = get_init_population_list()
    ipl = pd.DataFrame(init_population_list)
    ipl.to_csv('result/ipl.csv', index=False)

    # step2 取出最優的作為親代
    ipl = pd.read_csv('result/ipl.csv')
    tar_data = ipl
    best_score_list = []
    x_arr = []
    start_t = datetime.datetime.now()
    score = 0
    for count in range(iteration_num):
        _data, best_score = do_algorithm(tar_data, score)
        tar_data = _data
        x_arr.append(count + 1)

        try:
            print(best_score['fs'])
            score = float(best_score['fs'])
            if score is not None:
                if score < best_score_so_far['fs']:
                    best_score_list.append(score)
                else:
                    best_score_list.append(best_score_so_far['fs'])
        except Exception as e:
            print(e)
        result_df = pd.DataFrame(best_score_list, columns=['fr'])
        result_df.to_csv('result/result_ga.csv')
        print('第%s次迭代' % count)
        print('目前為止累計最好: %s' % best_score_so_far['fs'])
        print('本次最好的分數為: %s' % score)
    end_t = datetime.datetime.now()
    print('總共執行%s' % (end_t - start_t))
    d = pd.read_csv('result/result_ga.csv')
    x_arr = d['Unnamed: 0'].tolist()
    best_score_list = d['fr'].tolist()
    plt.plot(x_arr, best_score_list, color='b')
    plt.xlabel('iterate number')  # 設定x軸標題
    plt.xticks(x_arr, rotation='vertical')  # 設定x軸label以及垂直顯示
    plt.title('genetic algorithm')  # 設定圖表標題
    x_major_locator = MultipleLocator(100)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    now = datetime.datetime.now()
    s = datetime.datetime.strftime(now, '%Y%m%d%H%M')
    fn = s + '_' + 'cr' + str(crossover_rate) + '_' + 'mr' + str(mutation_rate)
    plt.savefig('result/r/' + fn + '.png')
    with open('result/r/' + fn + ".txt", "w") as txt_file:
        for item in best_score_list:
            txt_file.write("%s\n" % item)
    plt.show()


for i in range(round_num):
    process()
