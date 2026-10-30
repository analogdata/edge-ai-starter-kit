idf_component_register(
    SRCS "main.cc" "features.c" "model_data.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_driver_gpio esp_driver_i2s esp_timer)
