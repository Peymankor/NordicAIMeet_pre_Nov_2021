library(tidyverse)
library(lubridate)

elsport_2020 <- read.csv2("elspot-prices_2020_hourly_nok.csv", skip = 2, sep=",")

elsport_2021 <- read.csv2("elspot-prices_2021_hourly_nok.csv", skip = 2, sep=",")



elsport_2021_bergen <- elsport_2021 %>% select(c("...1","Hours","Bergen")) %>% 
  rename("day"="...1","price"="Bergen") %>% 
  mutate(day_date = dmy(day)) %>% 
  select(-c("day"))



elsport_2020_bergen <- elsport_2020 %>% select(c("...1","Hours","Bergen")) %>% 
rename("day"="...1","price"="Bergen") %>% 
  mutate(day_date = dmy(day)) %>% 
  select(-c("day"))


elsport_2020_bergen_1week_feb <- elsport_2020_bergen %>% 
  filter(day_date>"2020-05-30" & day_date<"2020-06-29")  %>%  
  mutate(Hours = seq(1,696))

elsport_2021_bergen_1week_feb <- elsport_2021_bergen %>% 
  filter(day_date>"2021-05-30" & day_date<"2021-06-29") %>%     
  mutate(Hours = seq(1,696))


elsport_2020_bergen_1week_feb_count_h <- elsport_2020_bergen_1week_feb %>% 
  mutate(hours = rep(seq(1,24),29))

elsport_2021_bergen_1week_feb_count_h <- elsport_2021_bergen_1week_feb %>% 
  mutate(hours = rep(seq(1,24),29))

ggplot(elsport_2020_bergen_1week_feb_count_h,aes(hours, price)) +
  geom_line() +
  facet_wrap(. ~ day_date)

ggplot(elsport_2021_bergen_1week_feb_count_h,aes(hours, price)) +
  geom_line() +
  facet_wrap(. ~ day_date)  

elsport_2020_bergen_1week_feb_count_h_lag <- 
  elsport_2020_bergen_1week_feb_count_h %>% mutate(price_chnage=lead(price)-price) %>% 
  drop_na() 


quantile_value <- quantile(elsport_2020_bergen_1week_feb_count_h_lag$price_chnage, 
                           probs = seq(0,1,0.01), names = FALSE)



data_price_change <- tibble(price_change_quantile = quantile_value,
                            quantiles = seq(0,1,0.01))

write_csv(data_price_change,"../Pypart_nordic_ai_2021/price_change_m.csv")


hist_price_2021 <- elsport_2021_bergen_1week_feb_count_h %>% 
  select(c("hours","price"))

write_csv(hist_price_2021,"../Pypart_nordic_ai_2021/hist_price_2021_m.csv")


powel_data <- read_delim("data_powell.csv")

powel_data_weel <- powel_data %>% filter(Time < 169) %>% 
  rename("price"="PJMRT ")

colnames(powel_data_weel)
ggplot(powel_data_weel, aes(Time, price)) +
  geom_line()


ggplot(elsport_2020_bergen_1week_feb, aes(Hours, price)) +
  geom_line()

ggplot(elsport_2021_bergen_1week_feb, aes(Hours, price)) +
  geom_line()



24*29
