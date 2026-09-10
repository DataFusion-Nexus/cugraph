/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/detail/effective_device_memory.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

TEST(EffectiveDeviceMemory, UnregisteredHandleUsesDeviceCapacity)
{
  raft::handle_t handle{};
  EXPECT_EQ(cugraph::detail::effective_device_memory(handle),
            handle.get_device_properties().totalGlobalMem);
}

TEST(EffectiveDeviceMemory, RegisteredProviderPreservesBoundaryValues)
{
  raft::handle_t handle{};
  std::size_t available{};
  cugraph::detail::register_effective_device_memory_provider(handle, [&available] { return available; });

  EXPECT_EQ(cugraph::detail::effective_device_memory(handle), 0);
  available = 1;
  EXPECT_EQ(cugraph::detail::effective_device_memory(handle), 1);
  available = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(cugraph::detail::effective_device_memory(handle), available);

  EXPECT_TRUE(cugraph::detail::unregister_effective_device_memory_provider(handle));
  EXPECT_THROW(cugraph::detail::effective_device_memory(handle), cugraph::logic_error);
  EXPECT_TRUE(cugraph::detail::release_effective_device_memory_registration(handle));
  EXPECT_EQ(cugraph::detail::effective_device_memory(handle),
            handle.get_device_properties().totalGlobalMem);
}

TEST(EffectiveDeviceMemory, RejectsInvalidRegistrationState)
{
  raft::handle_t handle{};
  EXPECT_THROW(cugraph::detail::register_effective_device_memory_provider(handle, {}),
               cugraph::logic_error);

  cugraph::detail::register_effective_device_memory_provider(
    handle, []() -> std::size_t { throw std::runtime_error{"provider failure"}; });
  EXPECT_THROW(cugraph::detail::effective_device_memory(handle), std::runtime_error);
  EXPECT_THROW(cugraph::detail::register_effective_device_memory_provider(handle, [] { return 1; }),
               cugraph::logic_error);
  EXPECT_TRUE(cugraph::detail::unregister_effective_device_memory_provider(handle));
  EXPECT_FALSE(cugraph::detail::unregister_effective_device_memory_provider(handle));
  EXPECT_TRUE(cugraph::detail::release_effective_device_memory_registration(handle));
  EXPECT_FALSE(cugraph::detail::release_effective_device_memory_registration(handle));
}

TEST(EffectiveDeviceMemory, UnregisterWaitsForActiveProviderCall)
{
  raft::handle_t handle{};
  std::mutex mutex{};
  std::condition_variable changed{};
  std::size_t call_count{};
  bool entered{};
  bool release{};
  cugraph::detail::register_effective_device_memory_provider(handle, [&] {
    std::unique_lock lock{mutex};
    ++call_count;
    if (call_count > 1) { return std::size_t{8}; }
    entered = true;
    changed.notify_all();
    changed.wait(lock, [&] { return release; });
    return std::size_t{7};
  });

  auto read = std::async(std::launch::async, [&] {
    return cugraph::detail::effective_device_memory(handle);
  });
  {
    std::unique_lock lock{mutex};
    changed.wait(lock, [&] { return entered; });
  }

  std::promise<void> unregister_started{};
  auto unregister_started_future = unregister_started.get_future();
  auto unregister = std::async(std::launch::async, [&] {
    unregister_started.set_value();
    return cugraph::detail::unregister_effective_device_memory_provider(handle);
  });
  unregister_started_future.get();
  bool tombstone_observed{};
  for (std::size_t attempt = 0; attempt < 100000 && !tombstone_observed; ++attempt) {
    try {
      static_cast<void>(cugraph::detail::effective_device_memory(handle));
    } catch (cugraph::logic_error const&) {
      tombstone_observed = true;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(tombstone_observed);
  EXPECT_EQ(unregister.wait_for(std::chrono::seconds{0}), std::future_status::timeout);
  {
    std::lock_guard lock{mutex};
    release = true;
  }
  changed.notify_all();

  EXPECT_EQ(read.get(), 7);
  EXPECT_TRUE(unregister.get());
  EXPECT_THROW(cugraph::detail::effective_device_memory(handle), cugraph::logic_error);
  EXPECT_TRUE(cugraph::detail::release_effective_device_memory_registration(handle));
}
