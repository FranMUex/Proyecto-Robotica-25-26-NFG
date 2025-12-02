# Sources set

LIST(APPEND SOURCES
  ../src/specificworker.cpp
        ../src/hungarian.cpp
        ../src/room_detector.cpp
        ../src/ransac_line_detector.cpp
        ../src/time_series_plotter.cpp
        ../src/door_detector.cpp
        ../src/qcustomplot.cpp
        ../src/pointcloud_center_estimator.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
        ../src/hungarian.h
        ../src/room_detector.h
        ../src/ransac_line_detector.h
        ../src/common_types.h
        ../src/time_series_plotter.h
        ../src/nominal_room.h
        ../src/door_detector.h
        ../src/image_processor.h
        ../src/qcustomplot.h
        ../src/pointcloud_center_estimator.h
)

INCLUDE($ENV{ROBOCOMP}/cmake/modules/opencv4.cmake)

SET(LIBS ${LIBS} tbb Qt6PrintSupport)

LIST(APPEND LIBS ${LIBS})

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-g3 -fmax-errors=1 -fno-char8_t)

