import gc
import hashlib
import json
import multiprocessing
import random as rd
import signal
import time
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from efficient_apriori import apriori
from fpgrowth_py import fpgrowth
from pycspade.helpers import spade
# consider using http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/ apriori implementation


LOG_PATH = 'log.txt'
REPORT_PATH = 'res/report.json'
DATASET_PATH = 'res/online_retail.csv'
CONSIDER_ITEM_QUANTITY = False  # may cause max recursion depth exception on Fpgrowth
FILTER_OUT_CLIENT_REPEATED_FOR_APRIORI_FPGROWTH = False  # avoid sequences of an item to itself
FILTER_OUT_CLIENT_REPEATED_FOR_SPADE = False  # avoid sequences of an item to itself
FORCE_SEQUENTIAL_EID = True
DEFAULT_CLIENT_ID = -666
DATASET_MAX_ROWS = None
MINIMUM_SUPPORT = 0.2  # closer to 0 mine everything, closer to 1 mine nothing
MINIMUM_CONFIDENCE = 0.5  # if 0.7, then all the itemsets must have at least 70% of confidence, e.g. the itemset [car, wheel] with confidence 0.9% -> at least 90% of people who bought cars bought wells
RUN_APRIORI = True
RUN_FPGROWTH = True
RUN_SPADE = True
RUN_PER_COUNTRY = True
RUN_PER_CLIENT = False  # does not make sense
ITEMSET_SIZE_GT = 1
MINIMUM_AMOUNT_OF_ENTRIES = 10
RUN_TIMEOUT_S = 180
TIMEOUT_METHOD = None  # None, 'signal', 'multiprocessing'
SPADE_BLOCKLIST = []
PLOT = False
BLOCK_PLOTS = True
KEEP_UNIDENTIFIED_CLIENT = False
LESS_FREQUENT_ITEMS_THRESHOLD = 120
FILTER_LEAST_SOLD_ITEMS_APRIORI_FPGROWTH = False
FILTER_LEAST_SOLD_ITEMS_SPADE = True
PRINT_SPADE_NULL_CONFID_LIFT_RESULTS=False

# I have huge outputs
def print_to_screen_and_file(py_print, log_path):
    def log_print(*args, **kwargs):
        py_print(*args, **kwargs)  # regular print
        try:
            with open(log_path, "a") as file:
                py_print(*args, file=file, **kwargs)
        except:
            py_print(f'Could not log to file {log_path}')

    return log_print


try:
    open(LOG_PATH, 'w').close()  # clear file
except:
    pass
print = print_to_screen_and_file(print, LOG_PATH)

# initialize seeds
rd.seed(time.time() * rd.random())
ANONYMIZATION_SEED = str(rd.random()).replace('.', '')  # this must be stored somewhere safe

# read dataset
df = pd.read_csv(DATASET_PATH)

unidentified_client_transactions = len(df[df['CustomerID'].isnull()]['InvoiceNo'].unique())
print(f'Transactions without registered client: {unidentified_client_transactions}')

# filter
df = df[(df['UnitPrice'] > 0) & (df['Quantity'] > 0)]
if KEEP_UNIDENTIFIED_CLIENT:
    df_bkp = df.copy()
    df = df.dropna(subset=['CustomerID'])
else:
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].apply(int)

# to store structured data
client_transactions = {}


# to make stuff sequential for the aug
class Sequentializer():
    def __init__(self):
        self.kv_dict = {}
        self.vk_list = []

    def addId(self, normal_id):
        if normal_id is None:
            raise Exception()
        if normal_id not in self.kv_dict:
            seq_id = len(self.vk_list)
            self.kv_dict[normal_id] = seq_id
            self.vk_list.append(normal_id)
            return seq_id
        return None

    def getSeqId(self, normal_id):
        return self.kv_dict.get(normal_id, None)

    def getNormalId(self, seq_id):
        if 0 <= seq_id < len(self.vk_list):
            return self.vk_list[seq_id]
        return None

    def getSeqIdOrAdd(self, normal_id):
        seq_id = self.addId(normal_id)
        if seq_id is None:
            seq_id = self.getSeqId(normal_id)
        return seq_id


# to anonymize the customers
def get_client_hash(client):
    salted_input = f'"People\'s dream will never die!", Blackbeard - {client} - {ANONYMIZATION_SEED}'
    client_id = hashlib.sha256(salted_input.encode('utf-8')).hexdigest()
    return str(client_id)


# statistical analysis
TOP_K = 20
print('Statistical analysis')

entries = df.shape[0]
print(f'Amount of dataset entries: {entries}')

amount_transactions = df['InvoiceNo'].nunique()
print(f'Distinct transactions: {amount_transactions}')

distinct_clients = df['CustomerID'].nunique() - 1
print(f'Distinct clients: {distinct_clients}')

amount_items = df['StockCode'].nunique()
print(f'Amount of goods for sale: {amount_items}')

amount_countries = df['Country'].nunique()
print(f'Distinct: {amount_countries}')
countries = df['Country'].unique()
for c in countries:
    print(f'\t{c}')

df.loc[:, 'TransactionPrice'] = df['UnitPrice'] * df['Quantity']
total_earnings = np.sum(df['TransactionPrice'])
print(f'Total earnings: ${round(total_earnings, 2)}')

if PLOT:
    grouped_clients = df.groupby('CustomerID')

    most_buying_clients_price = grouped_clients['TransactionPrice'].agg(np.sum).sort_values(ascending=False)
    ax = most_buying_clients_price.head(TOP_K).plot(kind='bar', title='Most buying clients (price)')
    ax.set_ylabel('Total value $')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    least_buying_clients_price = most_buying_clients_price.sort_values(ascending=True)
    ax = least_buying_clients_price.head(TOP_K).plot(kind='bar', title='Least buying clients (price)')
    ax.set_ylabel('Total value $')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    most_buying_clients_amount = grouped_clients['Quantity'].agg(np.sum).sort_values(ascending=False)
    ax = most_buying_clients_amount.head(TOP_K).plot(kind='bar', title='Most buying clients (amount)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    least_buying_clients_amount = most_buying_clients_amount.sort_values(ascending=True)
    ax = least_buying_clients_amount.head(TOP_K).plot(kind='bar', title='Least buying clients (amount)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

grouped_items = df.groupby('Description')

most_sold_items_amount = grouped_items['Quantity'].agg(np.sum).sort_values(ascending=False)

if PLOT:
    ax = most_sold_items_amount.head(TOP_K).plot(kind='bar', title='Most sold items')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

least_sold_items_amount = most_sold_items_amount.sort_values(ascending=True)
items_sold_less_than_x = least_sold_items_amount.loc[lambda x: x < LESS_FREQUENT_ITEMS_THRESHOLD]
least_sold_items_set = set()
print()
print(f'Items sold less than {LESS_FREQUENT_ITEMS_THRESHOLD} times ({len(items_sold_less_than_x)})')
for item, amount in items_sold_less_than_x.items():
    least_sold_items_set.add(item)
    print(f'\t{item}: {amount}')

if PLOT:
    ax = least_sold_items_amount.head(TOP_K).plot(kind='bar', title='Least sold items')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    most_earning_items = grouped_items['TransactionPrice'].agg(np.sum).sort_values(ascending=False)
    ax = most_earning_items.head(TOP_K).plot(kind='bar', title='Most earning items')
    ax.set_ylabel('Total value $')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    least_earning_items = most_earning_items.sort_values(ascending=True)
    ax = least_earning_items.head(TOP_K).plot(kind='bar', title='Least earning items')
    ax.set_ylabel('Total value $')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    df.loc[:, 'Year'] = pd.to_datetime(df['InvoiceDate']).dt.year
    df.loc[:, 'Month'] = pd.to_datetime(df['InvoiceDate']).dt.month
    grouped_year = df.sort_values('InvoiceDate').groupby('Year')

    earnings_per_year = grouped_year['TransactionPrice'].agg(np.sum)
    ax = earnings_per_year.plot(kind='bar', title='Earnings per year')
    ax.set_ylabel('Earnings $')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    sells_per_year = grouped_year['Quantity'].agg(np.sum)
    ax = sells_per_year.plot(kind='bar', title='Sold items per year')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    invoices_per_year = grouped_year['InvoiceNo'].nunique()
    ax = invoices_per_year.plot(kind='bar', title='Invoices per year')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    earnings_per_year = grouped_year['TransactionPrice'].agg(np.sum)
    ax = earnings_per_year.plot(kind='bar', title='Earnings per Year')
    ax.set_ylabel('Value $')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    grouped_date = df.sort_values('InvoiceDate').groupby(['Year', 'Month'])

    invoices_per_date = grouped_date['InvoiceNo'].unique().agg(np.size)
    ax = invoices_per_date.plot(kind='bar', title='Invoices per Month')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    earnings_per_date = grouped_date['TransactionPrice'].agg(np.sum)
    ax = earnings_per_date.plot(kind='bar', title='Earnings per Month')
    ax.set_ylabel('Value $')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    unique_clients_per_date = grouped_date['CustomerID'].unique().agg(np.size)
    ax = unique_clients_per_date.plot(kind='bar', title='Unique clients per Month')
    ax.set_ylabel('Clients')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    grouped_countries = df.groupby('Country')

    most_earnings_per_country = grouped_countries['TransactionPrice'].agg(np.sum).sort_values(ascending=False)
    ax = most_earnings_per_country.head(TOP_K).plot(kind='bar', title='Earnings per Country (Top)')
    ax.set_ylabel('Value $')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    least_earnings_per_country = most_earnings_per_country.sort_values(ascending=True)
    ax = least_earnings_per_country.head(TOP_K).plot(kind='bar', title='Earnings per Country (Bottom)')
    ax.set_ylabel('Value $')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    most_sells_per_country = grouped_countries['Quantity'].agg(np.sum).sort_values(ascending=False)
    ax = most_sells_per_country.head(TOP_K).plot(kind='bar', title='Sells per Country (Top)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    least_sells_per_country = most_sells_per_country.sort_values(ascending=True)
    ax = least_sells_per_country.head(TOP_K).plot(kind='bar', title='Sells per Country (Bottom)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    top_invoices_per_country = grouped_countries['InvoiceNo'].unique().agg(np.size).sort_values(ascending=False)
    ax = top_invoices_per_country.head(TOP_K).plot(kind='bar', title='Invoices per Country (Top)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    bottom_invoices_per_country = top_invoices_per_country.sort_values(ascending=True)
    ax = bottom_invoices_per_country.head(TOP_K).plot(kind='bar', title='Invoices per Country (Bottom)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    top_clients_per_country = grouped_countries['CustomerID'].unique().agg(np.size).sort_values(ascending=False)
    ax = top_clients_per_country.head(TOP_K).plot(kind='bar', title='Clients per Country (Top)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    bottom_clients_per_country = top_clients_per_country.sort_values(ascending=True)
    ax = bottom_clients_per_country.head(TOP_K).plot(kind='bar', title='Clients per Country (Bottom)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    top_distinct_items_per_country = grouped_countries['Description'].unique().agg(np.size).sort_values(ascending=False)
    ax = top_distinct_items_per_country.head(TOP_K).plot(kind='bar', title='Distinct items per Country (Top)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    bottom_distinct_items_per_country = top_distinct_items_per_country.sort_values(ascending=True)
    ax = bottom_distinct_items_per_country.head(TOP_K).plot(kind='bar', title='Distinct items per Country (Bottom)')
    ax.set_ylabel('Amount')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

    if not BLOCK_PLOTS:
        plt.show()

    for country, data in most_sells_per_country[:5].items():
        grouped_country_item = df[df['Country'] == country].groupby('Description')
        most_sold_items_per_each_most_sells_country = grouped_country_item['Quantity'].agg(np.sum).sort_values(
            ascending=False)
        ax = most_sold_items_per_each_most_sells_country.head(TOP_K).plot(kind='bar',
                                                                          title=f'Most sold items in {country}')
        ax.set_ylabel('Amount')
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)

    if not BLOCK_PLOTS:
        plt.show()

print()

# filter least sold items
FILTER_LEAST_SOLD_ITEMS_ALL_DF = False
if FILTER_LEAST_SOLD_ITEMS_APRIORI_FPGROWTH and FILTER_LEAST_SOLD_ITEMS_SPADE:
    df = df[~df['Description'].isin(least_sold_items_set)]
    FILTER_LEAST_SOLD_ITEMS_ALL_DF = True


# organize data / clear data
if KEEP_UNIDENTIFIED_CLIENT:
    df = df_bkp
earlier_timestamp = float('inf')
invoice_timestamps = []
least_sold_items_set_by_seq_id = set()
item_sequentializer = Sequentializer()  # cspade can't handle string items
for i, (idx, row) in enumerate(df.iterrows()):
    if pd.isna(row['UnitPrice']) or row['UnitPrice'] < 0 or pd.isna(row['Quantity']) or row['Quantity'] < 0 \
            or pd.isna(row['Description']) or pd.isna(row['StockCode']) or pd.isna(row['InvoiceNo']) \
            or pd.isna(row['Country']):
        df.drop(idx, inplace=True)
        continue
    # stock_code_to_item_name[row['StockCode']] = row['Description'] # no need to store stockcode, since it is string
    item_id = item_sequentializer.getSeqIdOrAdd(row['Description'].strip())
    if row['Description'] in least_sold_items_set:
        least_sold_items_set_by_seq_id.add(item_id)
    if pd.isna(row['CustomerID']):
        client = DEFAULT_CLIENT_ID
    else:
        client = get_client_hash(int(row['CustomerID']))
    if client not in client_transactions:
        client_transactions[client] = {}
    this_client_transactions = client_transactions[client]
    if row['InvoiceNo'] not in this_client_transactions:
        invoice_date = datetime.strptime(row['InvoiceDate'], '%d/%m/%y %H:%M')
        invoice_ts = int(invoice_date.timestamp())
        if FORCE_SEQUENTIAL_EID:
            invoice_timestamps.append(invoice_ts)
        earlier_timestamp = min(invoice_ts, earlier_timestamp)
        this_client_transactions[row['InvoiceNo']] = {'timestamp': invoice_ts, 'country': row['Country'].strip(),
                                                      'items': []}
    this_transaction = this_client_transactions[row['InvoiceNo']]
    if CONSIDER_ITEM_QUANTITY:
        this_transaction['items'].extend([item_id] * int(row['Quantity']))
    else:
        this_transaction['items'].append(item_id)

    if DATASET_MAX_ROWS is not None and i >= DATASET_MAX_ROWS:
        break

# free ram
del df
gc.collect()

if FORCE_SEQUENTIAL_EID:
    invoice_timestamps.sort()
    invoice_timestamps = {ts: (i + 1) for i, ts in enumerate(invoice_timestamps)}

# data for the algs
all_spade_data = {'All': []}
all_apriori_fpgrowth_data = {'All': []}

# parse data for the algorithms
filter_out_repeated = FILTER_OUT_CLIENT_REPEATED_FOR_SPADE or FILTER_OUT_CLIENT_REPEATED_FOR_APRIORI_FPGROWTH
client_sequentializer = Sequentializer()
for c_id, transactions in client_transactions.items():
    s_id = client_sequentializer.getSeqIdOrAdd(c_id)
    repeated_client_items = {'All': set()}
    for transaction in transactions.values():
        if FORCE_SEQUENTIAL_EID:
            simplified_ts = invoice_timestamps[transaction['timestamp']]
        else:
            simplified_ts = transaction['timestamp'] - earlier_timestamp  # huge values cause segfault on spade
        if RUN_PER_COUNTRY:
            if transaction['country'] not in all_spade_data:
                all_spade_data[transaction['country']] = []
                all_apriori_fpgrowth_data[transaction['country']] = []
            if transaction['country'] not in repeated_client_items:
                repeated_client_items[transaction['country']] = set()
        if RUN_PER_CLIENT:
            all_spade_data[f'Client {s_id}'] = []
            all_apriori_fpgrowth_data[f'Client {s_id}'] = []
        filtered_items = []
        filtered_items_country = []
        if filter_out_repeated:
            for client_item in transaction['items']:
                if client_item not in repeated_client_items['All']:
                    filtered_items.append(client_item)
                repeated_client_items['All'].add(client_item)
                if RUN_PER_COUNTRY:
                    if client_item not in repeated_client_items[transaction['country']]:
                        filtered_items_country.append(client_item)
                    repeated_client_items[transaction['country']].add(client_item)
        the_items = filtered_items if FILTER_OUT_CLIENT_REPEATED_FOR_SPADE else transaction['items']

        if len(the_items) > 0:
            all_spade_data['All'].append([s_id, simplified_ts, the_items])
            if FILTER_LEAST_SOLD_ITEMS_SPADE and not FILTER_LEAST_SOLD_ITEMS_ALL_DF:
                all_spade_data['All'][-1][2] = [spi for spi in all_spade_data['All'][-1][2] if
                                                spi not in least_sold_items_set_by_seq_id]
        if RUN_PER_COUNTRY:
            the_items = filtered_items_country if FILTER_OUT_CLIENT_REPEATED_FOR_SPADE else transaction['items']
            if len(the_items) > 0:
                all_spade_data[transaction['country']].append([s_id, simplified_ts, the_items])
                if FILTER_LEAST_SOLD_ITEMS_SPADE and not FILTER_LEAST_SOLD_ITEMS_ALL_DF:
                    all_spade_data[transaction['country']][-1][2] = [spi for spi in
                                                                     all_spade_data[transaction['country']][-1][2] if
                                                                     spi not in least_sold_items_set_by_seq_id]
        if RUN_PER_CLIENT:
            all_spade_data[f'Client {s_id}'].append([s_id, simplified_ts, transaction['items']])
            if FILTER_LEAST_SOLD_ITEMS_SPADE and not FILTER_LEAST_SOLD_ITEMS_ALL_DF:
                all_spade_data[transaction[f'Client {s_id}']][-1][2] = [spi for spi in
                                                                        all_spade_data[transaction[f'Client {s_id}']][
                                                                            -1][2] if
                                                                        spi not in least_sold_items_set_by_seq_id]

        items_tuple = tuple(filtered_items if FILTER_OUT_CLIENT_REPEATED_FOR_APRIORI_FPGROWTH else transaction['items'])
        items_tuple_country = tuple(
            filtered_items_country if FILTER_OUT_CLIENT_REPEATED_FOR_APRIORI_FPGROWTH else transaction['items'])
        if FILTER_LEAST_SOLD_ITEMS_APRIORI_FPGROWTH and not FILTER_LEAST_SOLD_ITEMS_ALL_DF:
            items_tuple = tuple([afpi for afpi in items_tuple if afpi not in least_sold_items_set_by_seq_id])
            items_tuple_country = tuple(
                [afpi for afpi in items_tuple_country if afpi not in least_sold_items_set_by_seq_id])
        if len(items_tuple) > 0:
            all_apriori_fpgrowth_data['All'].append(items_tuple)
        if RUN_PER_COUNTRY and len(items_tuple_country) > 0:
            all_apriori_fpgrowth_data[transaction['country']].append(items_tuple_country)
        if RUN_PER_CLIENT:
            all_apriori_fpgrowth_data[f'Client {s_id}'].append(transaction['items'])

del client_transactions
gc.collect()


def parse_apriori_results(apriori_result, sequentializer=None):
    res = {'itemsets': [], 'rules': []}
    itemsets, rules = apriori_result
    for k, v in itemsets.items():
        for k2, v2 in sorted(list(v.items()), key=lambda x: x[1], reverse=True):
            if sequentializer is not None:
                k2 = [sequentializer.getNormalId(x) for x in k2]
            res['itemsets'].append([list(k2), v2])
    rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
    for rule in rules_rhs:
        if sequentializer is not None:
            res['rules'].append(
                [list([sequentializer.getNormalId(x) for x in rule.lhs]),
                 list([sequentializer.getNormalId(x) for x in rule.rhs]),
                 [rule.confidence, rule.support, rule.lift, rule.conviction]])
        else:
            res['rules'].append(
                [list(rule.lhs), list(rule.rhs), [rule.confidence, rule.support, rule.lift, rule.conviction]])
    return res


def apriori_print(apriori_result, sequentializer=None, itemset_size_gt=0, name=''):
    printed = False
    printed_name = False
    itemsets, rules = apriori_result
    if len(itemsets) > 0:
        itemset_keys = list(itemsets.keys())
        itemset_keys.sort(reverse=True)
        for k in itemset_keys:
            if k > itemset_size_gt:
                v = itemsets[k]
                printed_inner = False
                for k2, v2 in sorted(list(v.items()), key=lambda x: x[1], reverse=True):
                    if sequentializer is not None:
                        k2 = [sequentializer.getNormalId(x) for x in k2]
                    if not printed and name != '':
                        print('=====')
                        print(name)
                        print('=====')
                        printed_name = True
                    if not printed:
                        print('Itemsets:')
                    if not printed_inner:
                        print(f'\tSize {k}:')
                    print(f'\t\t{list(k2)} ({v2})')
                    printed = True
                    printed_inner = True
        if printed:
            print()
    rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
    rules_rhs = sorted(rules_rhs, key=lambda rule: rule.lift)
    prev = printed
    printed = False
    if len(rules_rhs) > 0:
        for rule in rules_rhs:
            if sequentializer is not None:
                lhs = tuple([sequentializer.getNormalId(x) for x in rule.lhs])
                rhs = tuple([sequentializer.getNormalId(x) for x in rule.rhs])
            else:
                lhs = rule.lhs
                rhs = rule.rhs
            if not printed and not printed_name and name != '':
                print('=====')
                print(name)
                print('=====')
                printed_name = True
            if not printed:
                print('Rules:')
            print(
                f'\t{lhs} -> {rhs} (conf: {round(rule.confidence, 4)} supp: {round(rule.support, 4)} lift: {round(rule.lift, 4)} conv: {round(rule.conviction, 4)})')
            printed = True
    return printed or prev


def parse_fpgrowth_results(fpgrowth_result, sequentializer=None):
    res = {'itemsets': [], 'rules': []}
    if fpgrowth_result is None:
        return res
    itemsets, rules = fpgrowth_result

    for itemset in itemsets:
        if sequentializer is not None:
            itemset = [sequentializer.getNormalId(x) for x in itemset]
        res['itemsets'].append(list(itemset))
    for lhs, rhs, conf in rules:
        if sequentializer is not None:
            lhs = [sequentializer.getNormalId(x) for x in lhs]
            rhs = [sequentializer.getNormalId(x) for x in rhs]
        res['rules'].append([list(lhs), list(rhs), conf])
    return res


def fpgrowth_print(fpgrowth_result, sequentializer=None, itemset_size_gt=0, name=''):
    if fpgrowth_result is None:
        return False
    printed = False
    printed_name = False
    itemsets, rules = fpgrowth_result
    if len(itemsets) > 0:
        itemsets.sort(key=lambda x: len(x), reverse=True)
        for itemset in itemsets:
            if len(itemset) > itemset_size_gt:
                if sequentializer is not None:
                    itemset = [sequentializer.getNormalId(x) for x in itemset]

                if not printed and name != '':
                    print('=====')
                    print(name)
                    print('=====')
                    printed_name = True
                if not printed:
                    print('Itemsets:')
                print(f'\t{itemset} #{len(itemset)}')
                printed = True
        if printed:
            print()
    prev = printed
    printed = False
    if len(rules) > 0:
        rules.sort(key=lambda x: x[2], reverse=True)
        for lhs, rhs, conf in rules:
            if sequentializer is not None:
                lhs = [sequentializer.getNormalId(x) for x in lhs]
                rhs = [sequentializer.getNormalId(x) for x in rhs]
            if not printed and not printed_name and name != '':
                print('=====')
                print(name)
                print('=====')
                printed_name = True
            if not printed:
                print('Rules:')
            print(f'\t{lhs} -> {rhs} | Conf: {conf}')
            printed = True
    return printed or prev


def transform_spade_items(spade_result, sequentializer):
    for mined_obj in spade_result['mined_objects']:
        for item in mined_obj.items:
            item.elements = [sequentializer.getNormalId(el) for el in item.elements]


def parse_spade_results(spade_result, seq_occr_gt=0):
    res = []
    nseqs = spade_result['nsequences']
    for mined_object in spade_result['mined_objects']:
        if mined_object.noccurs > seq_occr_gt:
            res.append([list(map(str, mined_object.items)), mined_object.noccurs, mined_object.accum_occurs,
                        mined_object.noccurs / nseqs, mined_object.confidence, mined_object.lift])
    return res


def spade_print(spade_result, sequentializer=None, seq_occr_gt=0, name=''):
    # Code modified from the pycspade lib, in order to print to file
    printed = False
    if sequentializer is not None:
        transform_spade_items(spade_result, sequentializer)

    nseqs = spade_result['nsequences']
    spade_result['mined_objects'].sort(key=lambda x: (x.confidence if x.confidence else 0, x.lift if x.lift else 0,), reverse=True)
    for mined_object in spade_result['mined_objects']:
        if mined_object.noccurs <= seq_occr_gt:
            continue
        if not PRINT_SPADE_NULL_CONFID_LIFT_RESULTS and (not mined_object.confidence or not mined_object.lift):
            continue
        conf = 'N/A'
        lift = 'N/A'
        if mined_object.confidence:
            conf = '{:0.7f}'.format(mined_object.confidence)
        if mined_object.lift:
            lift = '{:0.7f}'.format(mined_object.lift)

        if not printed and name != '':
            print('=====')
            print(name)
            print('=====')
        if not printed:
            print(('{0:>9s} {1:>9s} {2:>9s} {3:>9s} {4:>9s} {5:>80s}'.format('Occurs', 'Accum', 'Support', 'Confid',
                                                                             'Lift',
                                                                             'Sequence')))
        print(('{0:>9d} {1:>9d} {2:>0.7f} {3:>9s} {4:>9s} {5:>80s} '.format(
            mined_object.noccurs,
            mined_object.accum_occurs,
            mined_object.noccurs / nseqs,
            conf,
            lift,
            '->'.join(list(map(str, mined_object.items))))))
        printed = True
    return printed


def get_spade_example_data():
    return [
        [1, 10, [3, 4]],
        [1, 15, [1, 2, 3]],
        [1, 20, [1, 2, 6]],
        [1, 25, [1, 3, 4, 6]],
        [2, 15, [1, 2, 6]],
        [2, 20, [5]],
        [3, 10, [1, 2, 6]],
        [4, 10, [4, 7, 8]],
        [4, 20, [2, 6]],
        [4, 25, [1, 7, 8]]
    ]


def get_apriori_fpgrowth_example_data():
    return [('eggs', 'bacon', 'soup'),
            ('eggs', 'bacon', 'apple'),
            ('soup', 'bacon', 'banana')]

def save_results_report(rep):
    try:
        with open(REPORT_PATH, 'w') as f:
            json.dump(rep, f)
    except Exception as e:
        print(e)


def spade_worker(a, b, c):
    spade_result = spade(data=a, support=b)
    c.append(spade_result)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)


def test():
    spade_result = spade(data=get_spade_example_data(), support=MINIMUM_SUPPORT)
    spade_print(spade_result)
    exit()


# test()

run_results = {'configs': {'ANONYMIZATION_SEED': ANONYMIZATION_SEED, 'DATASET_PATH': DATASET_PATH,
                           'CONSIDER_ITEM_QUANTITY': CONSIDER_ITEM_QUANTITY,
                           'FORCE_SEQUENTIAL_EID': FORCE_SEQUENTIAL_EID, 'DEFAULT_CLIENT_ID': DEFAULT_CLIENT_ID,
                           'DATASET_MAX_ROWS': DATASET_MAX_ROWS,
                           'MINIMUM_SUPPORT': MINIMUM_SUPPORT, 'MINIMUM_CONFIDENCE': MINIMUM_CONFIDENCE,
                           'RUN_PER_COUNTRY': RUN_PER_COUNTRY, 'RUN_PER_CLIENT': RUN_PER_CLIENT,
                           'ITEMSET_SIZE_GT': ITEMSET_SIZE_GT,
                           'MINIMUM_AMOUNT_OF_ENTRIES': MINIMUM_AMOUNT_OF_ENTRIES,
                           'FILTER_OUT_CLIENT_REPEATED_FOR_APRIORI_FPGROWTH': FILTER_OUT_CLIENT_REPEATED_FOR_APRIORI_FPGROWTH,
                           'FILTER_OUT_CLIENT_REPEATED_FOR_SPADE': FILTER_OUT_CLIENT_REPEATED_FOR_SPADE,
                           'RUN_TIMEOUT_S': RUN_TIMEOUT_S, 'SPADE_BLOCKLIST': SPADE_BLOCKLIST},
               'results': {},
               'sequentializers': {'items': item_sequentializer.kv_dict, 'clients': client_sequentializer.kv_dict}}
if RUN_APRIORI:
    print('APRIORI:')
    print()
    run_results['results']['Apriori'] = {}
    data_keys = list(all_apriori_fpgrowth_data.keys())
    data_keys.sort(key=lambda x: len(all_apriori_fpgrowth_data[x]))
    for idx, data_type in enumerate(data_keys):
        apriori_data = all_apriori_fpgrowth_data[data_type]
        if len(apriori_data) < MINIMUM_AMOUNT_OF_ENTRIES:
            continue
        name = ''
        if RUN_PER_COUNTRY or RUN_PER_CLIENT:
            name = f'{data_type} ({len(apriori_data)}):'

        start = perf_counter()
        apriori_result = apriori(apriori_data, min_support=MINIMUM_SUPPORT, min_confidence=MINIMUM_CONFIDENCE,
                                 output_transaction_ids=False)
        end = perf_counter()
        if apriori_print(apriori_result, item_sequentializer, ITEMSET_SIZE_GT, name):
            run_results['results']['Apriori'][data_type] = parse_apriori_results(apriori_result, item_sequentializer)
            print()
            print(f'Apriori algorithm took {end - start} seconds to run.')
            if idx != len(all_apriori_fpgrowth_data) - 1:
                print()
                print()
        save_results_report(run_results)

    print('======================================')
    print()
elif not RUN_FPGROWTH:
    for apriori_data in all_apriori_fpgrowth_data.values():
        del apriori_data
    del all_apriori_fpgrowth_data
    gc.collect()

if RUN_FPGROWTH:
    print('FPGrowth:')
    print()
    run_results['results']['FPGrowth'] = {}
    data_keys = list(all_apriori_fpgrowth_data.keys())
    data_keys.sort(key=lambda x: len(all_apriori_fpgrowth_data[x]))
    for idx, data_type in enumerate(data_keys):
        fpgrowth_data = all_apriori_fpgrowth_data[data_type]
        if len(fpgrowth_data) < MINIMUM_AMOUNT_OF_ENTRIES:
            continue
        name = ''
        if RUN_PER_COUNTRY or RUN_PER_CLIENT:
            name = f'{data_type} ({len(fpgrowth_data)}):'

        start = perf_counter()
        fpgrowth_result = fpgrowth(fpgrowth_data, minSupRatio=MINIMUM_SUPPORT, minConf=MINIMUM_CONFIDENCE)
        end = perf_counter()
        if fpgrowth_print(fpgrowth_result, item_sequentializer, ITEMSET_SIZE_GT, name):
            print()
            print(f'FPGrowth algorithm took {end - start} seconds to run.')
            run_results['results']['FPGrowth'][data_type] = parse_fpgrowth_results(fpgrowth_result, item_sequentializer)
            if idx != len(all_apriori_fpgrowth_data) - 1:
                print()
                print()
        save_results_report(run_results)

    print('======================================')
    print()
    del all_apriori_fpgrowth_data
    gc.collect()

if RUN_SPADE:
    print('SPADE:')
    print()
    run_results['results']['SPADE'] = {}
    data_keys = list(all_spade_data.keys())
    data_keys.sort(key=lambda x: len(all_spade_data[x]))
    for idx, data_type in enumerate(data_keys):
        spade_data = all_spade_data[data_type]
        if len(spade_data) < MINIMUM_AMOUNT_OF_ENTRIES or data_type in SPADE_BLOCKLIST:
            continue
        name = ''
        if RUN_PER_COUNTRY or RUN_PER_CLIENT:
            name = f'{data_type} ({len(spade_data)}):'
        start = perf_counter()
        spade_result = None
        last_e = None
        if TIMEOUT_METHOD is None:
            cur_t = 0
            max_tries = 2
            while spade_result is None and cur_t < max_tries:
                try:
                    spade_result = spade(data=spade_data, support=MINIMUM_SUPPORT)
                except Exception as e:
                    print()
                    print(e)
                    cur_t += 1
                    last_e = e
            if spade_result is None:
                if 'open data file' not in str(last_e):
                    raise last_e
                continue
        elif TIMEOUT_METHOD.lower() == 'multiprocessing':
            # I believe im missing if __name__ == main for this to work properly, so ill use signals
            spade_result = []
            p = multiprocessing.Process(target=spade_worker, name="Spade Worker",
                                        args=(spade_data, MINIMUM_SUPPORT, spade_result))
            p.start()
            p.join(RUN_TIMEOUT_S)
            if p.is_alive():
                p.terminate()
                p.join()
            if len(spade_result) > 0:
                spade_result = spade_result[0]
            else:
                continue
        elif TIMEOUT_METHOD.lower() == 'signal':
            # neither signal nor multiprocessor are working, ill use blocklist
            signal.alarm(RUN_TIMEOUT_S)
            try:
                spade_result = spade(data=spade_data, support=MINIMUM_SUPPORT)
                signal.alarm(0)  # disable alarm
            except TimeoutException:
                print('Aborting SPADE execution, reached time out')
            if spade_result is None:
                continue
        end = perf_counter()
        transform_spade_items(spade_result, item_sequentializer)
        if spade_print(spade_result, None, ITEMSET_SIZE_GT, name):
            run_results['results']['SPADE'][data_type] = parse_spade_results(spade_result, ITEMSET_SIZE_GT)
            print()
            print(f'SPADE algorithm took {end - start} seconds to run.')
            if idx != len(all_spade_data) - 1:
                print()
                print()
        save_results_report(run_results)

print()
