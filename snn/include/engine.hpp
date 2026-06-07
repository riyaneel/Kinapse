#pragma once

#define ALWAYS_INLINE __attribute__((always_inline)) inline

#define ENGINE_COLD_PATH __attribute__((cold))

#define ENGINE_HOT_PATH __attribute__((hot)) ALWAYS_INLINE
