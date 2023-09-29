import random
from datetime import datetime, timedelta
import pandas as pd

from make_data import make_data, profile_lst, start_date, order_size
from sklearn.utils import shuffle

capacity = 30
population_size = 30
crossover_rate = 0.7
crossover_count = 10
origin_count = 10
cut_len = 1
mutation_rate = 0.7
mutation_count = 5
pd.set_option('display.max_columns', None)


def init_data(data):
    result = []
    for pf in profile_lst:
        pf_df = data[data['profile'] == pf]
        pf_df = shuffle(pf_df)
        current_group = []
        pf_group = []
        if len(pf_df) == 0:
            continue
        last_df = pf_df.iloc[-1]
        if len(last_df) == 0:
            continue
        for idx, row in pf_df.iterrows():
            tar_df = data[data['order_id'] == row.order_id].iloc[0]
            if len(tar_df) == 0:
                continue
            tar_size = tar_df['sample_size']
            # 確認這次分組的內容總數
            current_size = data[data['order_id'].isin(current_group)]['sample_size'].sum()
            # 若最後一筆
            if last_df.order_id == tar_df.order_id:
                # 上一次剛好裝滿
                if current_size == capacity:
                    # print('目前裝滿, 新增一群')
                    pf_group.append(current_group)
                    result.extend(pf_group)
                    result.extend([[last_df.order_id]])
                    continue
                # 沒有空間了, 新增一群
                if (current_size + tar_size) > capacity:
                    # print('沒有空間了, 新增一群')
                    pf_group.append(current_group)
                    result.extend(pf_group)
                    result.extend([[last_df.order_id]])
                    continue
                else:
                    # print('有空間, 新增最後一群')
                    current_group.append(tar_df.order_id)
                    pf_group.append(current_group)
                    result.extend(pf_group)
                    continue

            else:
                # 若不是最後一筆
                # 若這次裝滿, 先裝入
                if current_size == capacity:
                    pf_group.append(current_group)
                    current_group = [tar_df.order_id]
                # 沒有空間了
                else:
                    if (current_size + tar_size) > capacity:
                        pf_group.append(current_group)
                        current_group = [tar_df.order_id]
                    else:
                        # 還有空間
                        current_group.append(tar_df.order_id)

    return result


def get_init_population(data):
    result = []
    # 取得初始母體
    for _ in range(population_size):
        init_chrom = init_data(data)
        for _gene in init_chrom:
            check_gene(_gene, data)
        random.shuffle(init_chrom)
        result.append(init_chrom)
    return result


def getOEE(gene, data):
    gene_df = data[data['order_id'].isin(gene)]
    count = 0
    for _, order in gene_df.iterrows():
        count += order.sample_size
    result = 1 - (count / capacity) + 0.05
    if count > capacity:
        print(gene_df)
        raise Exception
    return result


def fitness(chrom, data):
    next_open_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    result = 0
    chrom = remove_empty_lst(chrom)
    # print('fitness chromosome %s ' % chrom)
    for gene in chrom:
        gene_df = data[data['order_id'].isin(gene)].sort_values(by='arr_date')
        last_arr_order = gene_df.iloc[-1]
        last_arr_date = last_arr_order.arr_date
        last_arr_date = datetime.strptime(last_arr_date, '%Y-%m-%d %H:%M:%S')
        this_make_span = int(last_arr_order.make_span)
        if next_open_date < last_arr_date:
            next_open_date = last_arr_date + timedelta(hours=this_make_span)
        else:
            next_open_date = next_open_date + timedelta(hours=this_make_span)
        this_finish_date = next_open_date
        if type(this_finish_date) == str:
            this_finish_date = datetime.strptime(this_finish_date, '%Y-%m-%d %H:%M:%S')
        for order in gene:
            this_order = data[data['order_id'] == order].iloc[0]
            due_date = datetime.strptime(this_order.due_date, '%Y-%m-%d %H:%M:%S')
            delay = this_finish_date - due_date
            # 該訂單實際延遲時間(hr)
            delay_hr = round(delay.total_seconds() / 60 / 60)

            # 該訂單產品數量
            sample_size = this_order.sample_size
            # 該訂單權重
            this_weight = this_order.weight
            this_oee = getOEE(gene, data)

            if due_date > this_finish_date:
                delay_hr = 0

            lateness = delay_hr * this_weight * sample_size * this_oee
            if lateness < 0:
                print(this_order.due_date)
                print(this_finish_date)
                print('latness = %s * %s * %s * %s' % (delay_hr, this_weight, sample_size, this_oee))
                raise Exception
            result += lateness
    return result


def select(chromosomes, k, data):
    fs = []
    result = []
    for chromosome in chromosomes:
        score = fitness(chromosome, data)
        fs.append(score)
    s = []
    for f in fs:
        s.append(1 - (f / sum(fs)))

    for _ in range(k):
        _tar = random.choices(fs, weights=s, k=1)
        result.append(chromosomes[fs.index(_tar[0])])

    return result


def crossover(_pop, data):
    crossover_chromosome = []
    for _ in range(crossover_count):
        _parent_1, _parent_2 = select(_pop, 2, data)
        p1_len = len(_parent_1) - cut_len
        if p1_len < 1:
            p1_len = 1
        p2_len = len(_parent_2) - cut_len
        if p2_len < 1:
            p2_len = 1
        cut_index_1_start = random.randint(0, p1_len)
        cut_index_1_end = cut_index_1_start + cut_len
        cut_index_2_start = random.randint(0, p2_len)
        cut_index_2_end = cut_index_2_start + cut_len
        _parent_1_cut = _parent_1[cut_index_1_start:cut_index_1_end]
        _parent_2_cut = _parent_2[cut_index_2_start:cut_index_2_end]
        offspring_1 = _parent_1[:cut_index_1_start] + _parent_2_cut + _parent_1[cut_index_1_end:]
        offspring_2 = _parent_2[:cut_index_2_start] + _parent_1_cut + _parent_2[cut_index_2_end:]
        crossover_chromosome.append(offspring_1)
        crossover_chromosome.append(offspring_2)
    origin_chromosome = select(_pop, origin_count, data)
    return crossover_chromosome, origin_chromosome


def insert_chromosome(lose_lst, lose_chrom, data):
    no_set_order = []
    while len(lose_lst) > 0:
        lose = lose_lst.pop()
        lose_df = data[data['order_id'] == lose].iloc[0]
        lose_profile = lose_df.profile
        lose_sample_size = lose_df.sample_size
        for _gene in lose_chrom:
            gene_dfs = data[data['order_id'].isin(_gene)]
            gene_profile = gene_dfs.iloc[0].profile
            if gene_profile == lose_profile:
                gene_sample_size = gene_dfs['sample_size'].sum()
                if (int(gene_sample_size) + int(lose_sample_size)) > capacity:
                    # print('%s位置不夠' % lose)
                    continue
                else:
                    _gene.append(lose)
                    # print('插入 %s 於 %s' % (lose, _gene))
                    break
        else:
            no_set_order.append(lose)
    # print('沒有位置的訂單 %s' % no_set_order)
    return no_set_order


def remove_dup(remove_chromosome, _dup_lst):
    idx = 1
    col = ['order', 'group']
    rows = []
    for _gene in remove_chromosome:
        for order in _gene:
            rows.append([order, idx])
        idx += 1
    df = pd.DataFrame(rows, columns=col)
    drop_idx_lst = []
    for dup in _dup_lst:
        drop_idx = df[df['order'] == dup].index[-1]
        drop_idx_lst.append(drop_idx)
    _df = df.drop(drop_idx_lst)
    _new_chrom = []
    for i in range(idx):
        _new_gene = list(_df[_df['group'] == i]['order'].values)
        if len(_new_gene) > 0:
            _new_chrom.append(_new_gene)
    return _new_chrom


def fix(_chromosome, data):
    order_lst = []
    origin_lst = []
    for gene in _chromosome:
        for order in gene:
            order_lst.append(order)
    for i in range(order_size):
        origin_lst.append('o' + str(i + 1))
    lose_lst = [x for x in origin_lst if x not in order_lst]
    seen = set()
    dupe_lst = [x for x in order_lst if x in seen or seen.add(x)]
    # print('lose = %s' % lose_lst)
    # print('dup = %s' % dupe_lst)
    # 兩者均為0就不需要修復
    if (len(lose_lst) == 0) & (len(dupe_lst) == 0):
        return _chromosome
    else:
        # print('before remove dup chrom %s' % _chromosome)
        new_chrom = remove_dup(_chromosome, dupe_lst)
        # print('after remove dup chrom %s' % new_chrom)
        lose_df = data[data['order_id'].isin(lose_lst)].sort_values(by='arr_date')
        # 按照到站時間排序遺漏的訂單
        sorted_lose_lst = []
        for _, lost in lose_df.iterrows():
            sorted_lose_lst.append(lost.arr_date)

        no_set_order_lst = insert_chromosome(lose_lst, new_chrom, data)

        _lose_df = data[data['order_id'].isin(no_set_order_lst)].sort_values(by='arr_date')
        if len(_lose_df) > 0:
            r = init_data(_lose_df)
            new_chrom.extend(r)
        check_chromosome(new_chrom)
        for gene in new_chrom:
            check_gene(gene, data)
        new_chrom = remove_empty_lst(new_chrom)
        # print('final %s' % new_chrom)
        result = new_chrom
        return result


def remove_empty_lst(chromosome):
    result = []
    # 除去空基因
    for gene in chromosome:
        if len(gene) > 0:
            result.append(gene)
    return result


def check_gene(gene, data):
    gene_df = data[data['order_id'].isin(gene)]
    count = 0
    for _, order in gene_df.iterrows():
        count += order.sample_size
    if count > capacity:
        print('error gene_df %s' % gene_df)
        raise Exception


def check_chromosome(chrom):
    check = set()
    for gene in chrom:
        for order in gene:
            check.add(order)
    if len(check) != order_size:
        print('error chromosome %s' % chrom)
        raise Exception


def mutation(chromosome_lst, data):
    choices_chromosome = random.choices(chromosome_lst, k=round(mutation_rate * population_size))
    result = []
    for tar_chromosome in choices_chromosome:
        # print('before mutation %s' % tar_chromosome)
        for _ in range(mutation_count):
            mutation_group_idx = random.randint(round(len(tar_chromosome) / 2), len(tar_chromosome) - 1)
            tar_group_idx = random.randint(0, round(len(tar_chromosome) / 2) - 1)
            mutation_gene = tar_chromosome[mutation_group_idx]
            tar_gene = tar_chromosome[tar_group_idx]

            if len(mutation_gene) < 1:
                continue
            if len(tar_gene) < 1:
                continue
            mutation_order_df = data[data['order_id'].isin(mutation_gene)].sort_values(by='arr_date').iloc[-1]
            mutation_order = mutation_order_df.order_id
            mutation_profile = mutation_order_df.profile
            # mutation_gene = random.choices(tar_chromosome, k=1)
            # tar_gene = random.choices(tar_chromosome, k=1)
            # print('mutation gene %s' % mutation_gene)
            # print('tar gene %s' % tar_gene)
            # mutation_order_df = data[data['order_id'] == mutation_order].iloc[0]
            # mutation_order = random.choice(mutation_gene)
            # print('modify order %s' % mutation_order)

            gene_df = data[data['order_id'].isin(tar_gene)]
            tar_profile = gene_df.iloc[0].profile
            if tar_profile != mutation_profile:
                continue
            count = 0
            for _, order in gene_df.iterrows():
                count += order.sample_size
            if (count + mutation_order_df.sample_size) > capacity:
                continue
            if mutation_order in tar_gene:
                continue
            # print('start mutation %s from %s to %s' % (mutation_order, mutation_gene, tar_gene))
            mutation_gene.remove(mutation_order)
            tar_gene.append(mutation_order)
            # print('finish %s from %s to %s' % (mutation_order, mutation_gene, tar_gene))
        result.append(remove_empty_lst(tar_chromosome))
    return result


def do_algorithm(_pop, _data):
    non_feasible_chromosome_lst, origin_chromosome_lst = crossover(_pop, _data)
    modify_chromosome = []
    best_score = None
    best_chromosome = None
    for non_feasible_chromosome in non_feasible_chromosome_lst:
        # print('non_feasible_chromosome = %s' % non_feasible_chromosome)
        feasible_chromosome = fix(non_feasible_chromosome, _data)
        # print('fixed_chromosome = %s' % fixed_chromosome)
        modify_chromosome.append(feasible_chromosome)

    final_chromosome = mutation(modify_chromosome, _data)
    final_chromosome.extend(origin_chromosome_lst)
    for _final_chromosome in modify_chromosome:
        # print('new chromosome %s' % _final_chromosome)
        check_chromosome(_final_chromosome)
        fns = fitness(_final_chromosome, _data)
        # print('this fitness %s' % fns)
        if best_score is None:
            best_score = fns
        if best_chromosome is None:
            best_chromosome = _final_chromosome
        if fns < best_score:
            best_score = fns
            best_chromosome = _final_chromosome

    return modify_chromosome, best_score, best_chromosome


def exp():
    init_df = pd.read_csv('simulate_data.csv')
    _data = init_df
    population = get_init_population(init_df)
    current_population = population
    shuffle(current_population)
    best_score_so_far = None
    best_chromosome_so_far = None
    for num in range(3000):
        print('the %s iteration' % (num + 1))
        # print('current population %s' % current_population)
        new_population, _best_score_so_far, _best_chromosome_so_far = do_algorithm(current_population, _data)
        if best_score_so_far is None:
            best_score_so_far = _best_score_so_far
            best_chromosome_so_far = _best_chromosome_so_far
        if _best_score_so_far < best_score_so_far:
            best_score_so_far = _best_score_so_far
            best_chromosome_so_far = _best_chromosome_so_far
        print('best score so far %s' % best_score_so_far)
        # print('best chromosome so far %s' % best_chromosome_so_far)
        current_population = new_population


exp()
