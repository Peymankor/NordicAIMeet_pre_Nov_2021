def process_raw_price_data(file,params):
    
    import time as time
    DISC_TYPE = "FROM_CUM"
    #DISC_TYPE = "OTHER"

    print("Processing raw price data. Constructing price change list and cdf using {}".format(DISC_TYPE))
    tS = time.time()

    # load energy price data from the Excel spreadsheet
    raw_data = pd.read_excel(file, sheet_name="Raw Data")

    # look at data spanning a week
    data_selection = raw_data.iloc[0:params['T'], 0:5]

    # rename columns to remove spaces (otherwise we can't access them)
    cols = data_selection.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
    data_selection.columns = cols

    # sort prices in ascending order
    sort_by_price = data_selection.sort_values('PJM_RT_LMP')
    #print(sort_by_price.head())

    hist_price = np.array(data_selection['PJM_RT_LMP'].tolist())
    #print(hist_price[0])

    max_price = pd.DataFrame.max(sort_by_price['PJM_RT_LMP'])
    min_price = pd.DataFrame.min(sort_by_price['PJM_RT_LMP'])
    print("Min price {:.2f} and Max price {:.2f}".format(min_price,max_price))
    



    # sort prices in ascending order
    sort_by_price = data_selection.sort_values('PJM_RT_LMP')

    # calculate change in price and sort values of change in price in ascending order
    data_selection['Price_Shift'] = data_selection.PJM_RT_LMP.shift(1)
    data_selection['Price_Change'] = data_selection['PJM_RT_LMP'] - data_selection['Price_Shift']
    sort_price_change = data_selection.sort_values('Price_Change')
    

    # discretize change in price and obtain f(p) for each price change
    max_price_change = pd.DataFrame.max(sort_price_change['Price_Change'])
    min_price_change = pd.DataFrame.min(sort_price_change['Price_Change'])
    print("Min price change {:.2f} and Max price change {:.2f}".format(min_price_change,max_price_change))
    
    
    

    # there are 191 values for price change
    price_changes_sorted = sort_price_change['Price_Change'].tolist()
    # remove the last NaN value
    price_changes_sorted.pop()

    if DISC_TYPE == "FROM_CUM":
    # discretize price change  by interpolating from cumulative distribution
        xp = price_changes_sorted
        fp = np.arange(len(price_changes_sorted) - 1) / (len(price_changes_sorted) - 1)
        cum_fn = np.append(fp, 1)

        # obtain 30 discrete prices
        discrete_price_change_cdf = np.linspace(0, 1, params['nPriceChangeInc'])
        discrete_price_change_list = []
        for i in discrete_price_change_cdf:
            interpolated_point = np.interp(i, cum_fn, xp)
            discrete_price_change_list.append(interpolated_point)
    else:
        price_change_range = max_price_change - min_price_change
        price_change_increment = price_change_range / params['nPriceChangeInc']
        discrete_price_change = np.arange(min_price_change, max_price_change, price_change_increment)
        discrete_price_change_list = list(np.append(discrete_price_change, max_price_change))


        f_p = np.arange(len(price_changes_sorted) - 1) / (len(price_changes_sorted) - 1)
        cum_fn = np.append(f_p, 1)
        discrete_price_change_cdf = []
        for c in discrete_price_change_list:
            interpolated_point = np.interp(c, price_changes_sorted, cum_fn)
            discrete_price_change_cdf.append(interpolated_point)

    price_changes_sorted = np.array(price_changes_sorted)
    discrete_price_change_list = np.array(discrete_price_change_list)
    discrete_price_change_cdf = np.array(discrete_price_change_cdf)
    discrete_price_change_pdf = discrete_price_change_cdf - shift(discrete_price_change_cdf,1,cval=0)

    mean_price_change = np.dot(discrete_price_change_list,discrete_price_change_pdf)

    #print("discrete_price_change_list ",discrete_price_change_list)
    #print("discrete_price_change_cdf",discrete_price_change_cdf)
    #print("discrete_price_change_pdf",discrete_price_change_pdf)


    print("Finishing processing raw price data in {:.2f} secs. Expected price change is {:.2f}. Hist_price len is {}".format(time.time()-tS,mean_price_change,len(hist_price)))
    #input("enter any key to continue...") 

    exog_params = {'hist_price':hist_price,"price_changes_sorted":price_changes_sorted,"discrete_price_change_list":discrete_price_change_list,"discrete_price_change_cdf":discrete_price_change_cdf}

    return exog_params