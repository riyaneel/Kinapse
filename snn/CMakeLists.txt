add_library(core_engine_lib INTERFACE)

target_include_directories(core_engine_lib INTERFACE
		include
)

target_link_libraries(core_engine_lib
		INTERFACE core_compiler_flags
)
