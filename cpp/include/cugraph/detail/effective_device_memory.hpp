/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <raft/core/handle.hpp>

#include <cstddef>
#include <functional>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

using effective_device_memory_provider_t = std::function<std::size_t()>;

std::size_t effective_device_memory(raft::handle_t const& handle);

void register_effective_device_memory_provider(
  raft::handle_t const& handle, effective_device_memory_provider_t provider);

bool unregister_effective_device_memory_provider(raft::handle_t const& handle) noexcept;

bool release_effective_device_memory_registration(raft::handle_t const& handle) noexcept;

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph
