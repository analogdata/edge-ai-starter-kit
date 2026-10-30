cmake_minimum_required(VERSION 3.16)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/main/model_data.cc")
    message(FATAL_ERROR
        "main/model_data.cc not found. Train the model first:\n"
        "    cd ../train && pip install -r requirements.txt && python train.py\n"
        "(train.py writes the model into firmware/main for you.)")
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(edgeai_kws)
