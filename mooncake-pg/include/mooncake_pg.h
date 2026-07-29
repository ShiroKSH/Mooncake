#ifndef MOONCAKE_PG_H_
#define MOONCAKE_PG_H_

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define MOONCAKE_PG_EXPORT __attribute__((visibility("default")))
#else
#define MOONCAKE_PG_EXPORT
#endif

#define MOONCAKE_PG_MAX_RANKS 64
#define MOONCAKE_PG_COORDINATOR_ADDRESS_BYTES 256
#define MOONCAKE_PG_MAX_ERROR_STRING 256
#define MOONCAKE_PG_CONFIG_UNDEF_INT INT_MIN
#define MOONCAKE_PG_DEFAULT_COLLECTIVE_TIMEOUT_US 10000000ULL
#define MOONCAKE_PG_DEFAULT_P2P_TIMEOUT_US 10000000LL
#define MOONCAKE_PG_DEFAULT_FAULT_RECONCILIATION_WINDOW_US 30000000LL

#define MOONCAKE_PG_COMM_CONFIG_MAGIC 0xcafe4d50u
#define MOONCAKE_PG_COMM_CONFIG_VERSION 1u

typedef struct mooncakePgContext* mooncakePgContext_t;
typedef struct mooncakePgComm* mooncakePgComm_t;
typedef struct mooncakePgCompletion* mooncakePgCompletion_t;
/* A null handle selects the communicator device's legacy/default stream. */
typedef void* mooncakePgStream_t;

typedef enum mooncakePgResult {
    mooncakePgSuccess = 0,
    mooncakePgSystemError = 1,
    mooncakePgInternalError = 2,
    mooncakePgInvalidArgument = 3,
    mooncakePgTimeout = 4,
} mooncakePgResult_t;

typedef enum mooncakePgDataType {
    mooncakePgUint8 = 0,
    mooncakePgInt8 = 1,
    mooncakePgInt16 = 2,
    mooncakePgInt32 = 3,
    mooncakePgInt64 = 4,
    mooncakePgFloat32 = 5,
    mooncakePgFloat64 = 6,
    mooncakePgBool = 7,
    mooncakePgBfloat16 = 8,
} mooncakePgDataType_t;

typedef enum mooncakePgReduceOp {
    mooncakePgSum = 0,
    mooncakePgAvg = 1,
    mooncakePgProduct = 2,
    mooncakePgMin = 3,
    mooncakePgMax = 4,
} mooncakePgReduceOp_t;

typedef enum mooncakePgDeviceType {
    mooncakePgDeviceCpu = 0,
    mooncakePgDeviceGpu = 1,
} mooncakePgDeviceType_t;

typedef enum mooncakePgIdResolvePolicy {
    mooncakePgIdResolveCreateOrAttach = 0,
    mooncakePgIdResolveAttachOrExtend = 1,
} mooncakePgIdResolvePolicy_t;

typedef struct mooncakePgCoordinatorAddress {
    char internal[MOONCAKE_PG_COORDINATOR_ADDRESS_BYTES];
} mooncakePgCoordinatorAddress_t;

typedef struct mooncakePgCommConfig {
    size_t structSize;
    unsigned int magic;
    unsigned int version;
    const char* groupId;
    int rank;
    int size;
    int maxGroupSize;
    const int32_t* globalRanks;
    size_t globalRankCount;
    mooncakePgDeviceType_t deviceType;
    int deviceIndex;
    mooncakePgIdResolvePolicy_t idResolvePolicy;
    int autoDeactivateOnFailure;
    int autoSyncOnFailure;
    /* Optional caller-owned mirror of the communicator's active ranks. */
    int32_t* activeRanksMirror;
    size_t activeRanksMirrorCount;
    int activeRanksMirrorIsDevice;
} mooncakePgCommConfig_t;

#define MOONCAKE_PG_COMM_CONFIG_INITIALIZER \
    {sizeof(mooncakePgCommConfig_t),        \
     MOONCAKE_PG_COMM_CONFIG_MAGIC,         \
     MOONCAKE_PG_COMM_CONFIG_VERSION,       \
     NULL,                                  \
     MOONCAKE_PG_CONFIG_UNDEF_INT,          \
     MOONCAKE_PG_CONFIG_UNDEF_INT,          \
     MOONCAKE_PG_CONFIG_UNDEF_INT,          \
     NULL,                                  \
     0,                                     \
     mooncakePgDeviceGpu,                   \
     MOONCAKE_PG_CONFIG_UNDEF_INT,          \
     mooncakePgIdResolveCreateOrAttach,     \
     1,                                     \
     1,                                     \
     NULL,                                  \
     0,                                     \
     0}

typedef enum mooncakePgProposalStatus {
    mooncakePgProposalRejected = 0,
    mooncakePgProposalApplied = 1,
    mooncakePgProposalAppliedWithDroppedRanks = 2,
} mooncakePgProposalStatus_t;

typedef struct mooncakePgProposalResponse {
    mooncakePgProposalStatus_t status;
    uint64_t newEpoch;
    size_t droppedRankCount;
    int32_t droppedRanks[MOONCAKE_PG_MAX_RANKS];
    char rejectReason[MOONCAKE_PG_MAX_ERROR_STRING];
} mooncakePgProposalResponse_t;

typedef enum mooncakePgSyncAfterFailureStatus {
    mooncakePgSyncReconciled = 0,
    mooncakePgSyncNoPending = 1,
    mooncakePgSyncRejected = 2,
} mooncakePgSyncAfterFailureStatus_t;

typedef struct mooncakePgSyncAfterFailureResponse {
    mooncakePgSyncAfterFailureStatus_t status;
    char rejectReason[MOONCAKE_PG_MAX_ERROR_STRING];
} mooncakePgSyncAfterFailureResponse_t;

MOONCAKE_PG_EXPORT const char* mooncakePgGetErrorString(
    mooncakePgResult_t result);
MOONCAKE_PG_EXPORT const char* mooncakePgGetLastError(void);


MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgContextCreate(mooncakePgContext_t* context);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgContextInitialize(
    mooncakePgContext_t context, int globalRank, int maxWorldSize);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgContextLaunchCoordinator(
    mooncakePgContext_t context,
    mooncakePgCoordinatorAddress_t* coordinatorAddress);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgContextConnectCoordinator(
    mooncakePgContext_t context,
    const mooncakePgCoordinatorAddress_t* coordinatorAddress);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgContextSetHostIp(mooncakePgContext_t context, const char* hostIp);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgContextSetTransferEngine(
    mooncakePgContext_t context, void* transferEngine);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgContextSetDeviceFilter(
    mooncakePgContext_t context, const char* const* filters,
    size_t filterCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgContextSetCollectiveTimeout(
    mooncakePgContext_t context, size_t timeoutUs);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgContextSetP2PTimeout(mooncakePgContext_t context, int64_t timeoutUs);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgContextSetFaultReconciliationWindow(mooncakePgContext_t context,
                                              int64_t timeoutUs);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgContextDestroy(mooncakePgContext_t context);

MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgCommCreate(
    mooncakePgContext_t context, const mooncakePgCommConfig_t* config,
    mooncakePgComm_t* comm);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCommDestroy(mooncakePgComm_t comm);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCommGetRank(mooncakePgComm_t comm, int* rank);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCommGetSize(mooncakePgComm_t comm, int* size);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCommGetMaxGroupSize(mooncakePgComm_t comm, int* maxGroupSize);

MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgBroadcastGpu(const void* sendBuffer, void* recvBuffer, size_t count,
                       mooncakePgDataType_t dataType, int root,
                       mooncakePgComm_t comm, mooncakePgStream_t stream,
                       int32_t* failedRanksHint, size_t failedRanksHintCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgAllReduceGpu(
    const void* sendBuffer, void* recvBuffer, size_t count,
    mooncakePgDataType_t dataType, mooncakePgReduceOp_t reduceOp,
    mooncakePgComm_t comm, mooncakePgStream_t stream, int32_t* failedRanksHint,
    size_t failedRanksHintCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgAllGatherGpu(const void* sendBuffer, void* recvBuffer, size_t count,
                       mooncakePgDataType_t dataType, mooncakePgComm_t comm,
                       mooncakePgStream_t stream, int32_t* failedRanksHint,
                       size_t failedRanksHintCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgReduceScatterGpu(
    const void* sendBuffer, void* recvBuffer, size_t count,
    mooncakePgDataType_t dataType, mooncakePgReduceOp_t reduceOp,
    mooncakePgComm_t comm, mooncakePgStream_t stream, int32_t* failedRanksHint,
    size_t failedRanksHintCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgAllToAllGpu(const void* sendBuffer, void* recvBuffer, size_t count,
                      mooncakePgDataType_t dataType, mooncakePgComm_t comm,
                      mooncakePgStream_t stream, int32_t* failedRanksHint,
                      size_t failedRanksHintCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgReduceGpu(
    const void* sendBuffer, void* recvBuffer, size_t count,
    mooncakePgDataType_t dataType, mooncakePgReduceOp_t reduceOp, int root,
    mooncakePgComm_t comm, mooncakePgStream_t stream, int32_t* failedRanksHint,
    size_t failedRanksHintCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgGatherGpu(const void* sendBuffer, void* recvBuffer, size_t count,
                    mooncakePgDataType_t dataType, int root,
                    mooncakePgComm_t comm, mooncakePgStream_t stream,
                    int32_t* failedRanksHint, size_t failedRanksHintCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgScatterGpu(const void* sendBuffer, void* recvBuffer, size_t count,
                     mooncakePgDataType_t dataType, int root,
                     mooncakePgComm_t comm, mooncakePgStream_t stream,
                     int32_t* failedRanksHint, size_t failedRanksHintCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgBarrierGpu(mooncakePgComm_t comm, mooncakePgStream_t stream,
                     int32_t* failedRanksHint, size_t failedRanksHintCount);

MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgBroadcastCpu(
    const void* sendBuffer, void* recvBuffer, size_t count,
    mooncakePgDataType_t dataType, int root, mooncakePgComm_t comm,
    int32_t* failedRanksHint, size_t failedRanksHintCount,
    mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgAllReduceCpu(
    const void* sendBuffer, void* recvBuffer, size_t count,
    mooncakePgDataType_t dataType, mooncakePgReduceOp_t reduceOp,
    mooncakePgComm_t comm, int32_t* failedRanksHint,
    size_t failedRanksHintCount, mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgAllGatherCpu(const void* sendBuffer, void* recvBuffer, size_t count,
                       mooncakePgDataType_t dataType, mooncakePgComm_t comm,
                       int32_t* failedRanksHint, size_t failedRanksHintCount,
                       mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgReduceScatterCpu(
    const void* sendBuffer, void* recvBuffer, size_t count,
    mooncakePgDataType_t dataType, mooncakePgReduceOp_t reduceOp,
    mooncakePgComm_t comm, int32_t* failedRanksHint,
    size_t failedRanksHintCount, mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgAllToAllCpu(const void* sendBuffer, void* recvBuffer, size_t count,
                      mooncakePgDataType_t dataType, mooncakePgComm_t comm,
                      int32_t* failedRanksHint, size_t failedRanksHintCount,
                      mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgReduceCpu(
    const void* sendBuffer, void* recvBuffer, size_t count,
    mooncakePgDataType_t dataType, mooncakePgReduceOp_t reduceOp, int root,
    mooncakePgComm_t comm, int32_t* failedRanksHint,
    size_t failedRanksHintCount, mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgGatherCpu(
    const void* sendBuffer, void* recvBuffer, size_t count,
    mooncakePgDataType_t dataType, int root, mooncakePgComm_t comm,
    int32_t* failedRanksHint, size_t failedRanksHintCount,
    mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgScatterCpu(
    const void* sendBuffer, void* recvBuffer, size_t count,
    mooncakePgDataType_t dataType, int root, mooncakePgComm_t comm,
    int32_t* failedRanksHint, size_t failedRanksHintCount,
    mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgBarrierCpu(
    mooncakePgComm_t comm, int32_t* failedRanksHint,
    size_t failedRanksHintCount, mooncakePgCompletion_t* completion);

MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgSendGpu(
    const void* sendBuffer, size_t count, mooncakePgDataType_t dataType,
    int peer, mooncakePgComm_t comm, mooncakePgStream_t stream,
    int32_t* failedRanksHint, size_t failedRanksHintCount,
    mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgRecvGpu(
    void* recvBuffer, size_t count, mooncakePgDataType_t dataType, int peer,
    mooncakePgComm_t comm, mooncakePgStream_t stream, int32_t* failedRanksHint,
    size_t failedRanksHintCount, mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgSendCpu(
    const void* sendBuffer, size_t count, mooncakePgDataType_t dataType,
    int peer, mooncakePgComm_t comm, int32_t* failedRanksHint,
    size_t failedRanksHintCount, mooncakePgCompletion_t* completion);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgRecvCpu(
    void* recvBuffer, size_t count, mooncakePgDataType_t dataType, int peer,
    mooncakePgComm_t comm, int32_t* failedRanksHint,
    size_t failedRanksHintCount, mooncakePgCompletion_t* completion);


MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgCompletionIsCompleted(
    mooncakePgCompletion_t completion, int* completed);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCompletionWait(mooncakePgCompletion_t completion, int64_t timeoutUs);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCompletionDestroy(mooncakePgCompletion_t completion);

MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgCommGetActiveRanks(
    mooncakePgComm_t comm, int32_t* activeRanks, size_t rankCount);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCommGetPeerState(mooncakePgComm_t comm, const int32_t* ranks,
                           size_t rankCount, int32_t* peerStates);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgCommActivateRanks(
    mooncakePgComm_t comm, const int32_t* ranks, size_t rankCount,
    mooncakePgProposalResponse_t* response);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgCommDeactivateRanks(
    mooncakePgComm_t comm, const int32_t* ranks, size_t rankCount,
    mooncakePgProposalResponse_t* response);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgCommJoin(mooncakePgComm_t comm);
MOONCAKE_PG_EXPORT mooncakePgResult_t mooncakePgCommSyncAfterFailure(
    mooncakePgComm_t comm, mooncakePgSyncAfterFailureResponse_t* response);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCommGetEpoch(mooncakePgComm_t comm, uint64_t* epoch);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCommGetNumSyncedRanks(mooncakePgComm_t comm, int* count);
MOONCAKE_PG_EXPORT mooncakePgResult_t
mooncakePgCommGetPreferredHca(mooncakePgComm_t comm, const char* location,
                              char* hcaBuf, size_t hcaBufSize);

#ifdef __cplusplus
}
#endif

#endif  // MOONCAKE_PG_H_
