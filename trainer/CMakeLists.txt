include_directories(
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(snn_train_bin
		src/SnnTrainer.cpp
		src/main.cpp
)

target_link_libraries(snn_train_bin PRIVATE
		core_compiler_flags
		core_engine_lib
		Threads::Threads
)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if (ipo_supported)
	set_property(TARGET snn_train_bin PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()
