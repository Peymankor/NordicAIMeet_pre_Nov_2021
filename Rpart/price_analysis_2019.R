library(tidyverse)
library(lubridate)
library(patchwork)
elsport_2019 <- read.csv2("elspot-prices_2019_hourly_nok.csv", skip = 2, sep = ",")

elsport_2019_bergen <- elsport_2019 %>% select(c("X","Hours","Bergen")) %>% 
  rename("price"="Bergen") %>% 
  mutate(day_date = dmy(X)) %>%
  mutate(week=lubridate::week(day_date))
  select(-c("X"))


elsport_2020 <- read.csv2("elspot-prices_2020_hourly_nok.csv", skip = 2, sep = ",")

elsport_2020_bergen <- elsport_2020 %>% select(c("X","Hours","Bergen")) %>% 
  rename("price"="Bergen") %>% 
  mutate(day_date = dmy(X)) %>% 
  select(-c("X"))


elsport_2020_bergen_1week_feb <- elsport_2020_bergen %>% 
  filter(day_date>"2020-08-30" & day_date<"2020-09-29") %>%     
  mutate(Hours = seq(1,696)) %>% 
  mutate(weeks = rep(seq(1,)))

elsport_2019_bergen_1week_feb <- elsport_2019_bergen %>% 
  filter(day_date>"2019-08-30" & day_date<"2019-09-29") %>%         
  mutate(Hours = seq(1,696))



ggplot(elsport_2020_bergen_1week_feb, aes(Hours, price)) +
  geom_line()

ggplot(elsport_2019_bergen_1week_feb, aes(day_date, price)) +
  geom_line()



p1 <- ggplot(elsport_2019_bergen_1week_feb,aes(Hours,price)) +
  geom_line() + 
  ggtitle("Plot of spot electricty price, 2019 ") +
  xlab("Hours in month") + 
  ylab("Price, NOK/MWH  ")
  

p2 <- ggplot(elsport_2020_bergen_1week_feb,aes(Hours,price)) +
  geom_line() +
  ggtitle("Plot of spot electricty price, 2020 ") +
  xlab("Hours in month") + 
  ylab("Price, NOK/MWH  ")

p1 + p2

