/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/detail/effective_device_memory.hpp>
#include <cugraph/utilities/error.hpp>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace cugraph::detail {
namespace {

struct provider_entry_t {
  std::optional<effective_device_memory_provider_t> provider{};
  std::size_t active_calls{};
};

using provider_map_t =
  std::unordered_map<raft::handle_t const*, std::shared_ptr<provider_entry_t>>;

provider_map_t& providers()
{
  static provider_map_t value{};
  return value;
}

std::mutex& providers_mutex()
{
  static std::mutex value{};
  return value;
}

std::condition_variable& providers_changed()
{
  static std::condition_variable value{};
  return value;
}

class provider_call_guard_t {
 public:
  explicit provider_call_guard_t(std::shared_ptr<provider_entry_t> entry)
    : entry_(std::move(entry))
  {
  }

  provider_call_guard_t(provider_call_guard_t const&)            = delete;
  provider_call_guard_t& operator=(provider_call_guard_t const&) = delete;

  ~provider_call_guard_t()
  {
    std::lock_guard lock{providers_mutex()};
    --entry_->active_calls;
    providers_changed().notify_all();
  }

 private:
  std::shared_ptr<provider_entry_t> entry_;
};

}  // namespace

std::size_t effective_device_memory(raft::handle_t const& handle)
{
  std::shared_ptr<provider_entry_t> entry{};
  effective_device_memory_provider_t provider{};
  {
    std::lock_guard lock{providers_mutex()};
    auto const found = providers().find(&handle);
    if (found == providers().end()) { return handle.get_device_properties().totalGlobalMem; }
    entry = found->second;
    CUGRAPH_EXPECTS(entry->provider.has_value(),
                    "effective device memory provider is missing for a registered handle");
    provider = *entry->provider;
    ++entry->active_calls;
  }
  // Keep the provider owner alive until the callback has returned, including
  // when the callback raises an invariant error.
  provider_call_guard_t guard{std::move(entry)};
  return provider();
}

void register_effective_device_memory_provider(
  raft::handle_t const& handle, effective_device_memory_provider_t provider)
{
  CUGRAPH_EXPECTS(static_cast<bool>(provider), "effective device memory provider is empty");
  std::lock_guard lock{providers_mutex()};
  auto entry              = std::make_shared<provider_entry_t>();
  entry->provider         = std::move(provider);
  auto const [_, inserted] = providers().emplace(&handle, std::move(entry));
  CUGRAPH_EXPECTS(inserted, "effective device memory provider is already registered");
}

bool unregister_effective_device_memory_provider(raft::handle_t const& handle) noexcept
{
  std::unique_lock lock{providers_mutex()};
  auto const found = providers().find(&handle);
  if (found == providers().end() || !found->second->provider.has_value()) { return false; }
  auto entry = found->second;
  entry->provider.reset();
  providers_changed().wait(lock, [&entry] { return entry->active_calls == 0; });
  return true;
}

bool release_effective_device_memory_registration(raft::handle_t const& handle) noexcept
{
  std::lock_guard lock{providers_mutex()};
  auto const found = providers().find(&handle);
  if (found == providers().end() || found->second->provider.has_value() || found->second->active_calls != 0) {
    return false;
  }
  providers().erase(found);
  return true;
}

}  // namespace cugraph::detail
