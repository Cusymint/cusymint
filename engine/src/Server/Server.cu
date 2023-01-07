#include "Server.cuh"

#include "ResponseBuilder.cuh"
#include "SolverProcessManager.cuh"
#include "Logger.cuh"
#include "Parser/Parser.cuh"
#include "Utils/CompileConstants.cuh"

#include <fmt/core.h>

// https://mongoose.ws/tutorials/json-rpc-over-websocket/

// Use of global variables is required, since mongoose callbacks are meant to be C functions.

static struct mg_rpc* s_rpc_head = NULL;
static CachedParser* global_cached_parser = NULL;

static void interpret(struct mg_rpc_req* r) {
    auto input = mg_json_get_str(r->frame, "$.params.input");

    if (input == NULL) {
        Logger::print("[Server] Interpret couldn't find input\n");
        mg_rpc_err(r, 400, "Missing input");
        return;
    }

    Logger::print("[Server] Interpret input {}\n", input);
    auto response_builder = ResponseBuilder(); 

    try {
        auto parser_result = global_cached_parser->parse(input);
        response_builder.set_input(parser_result);
    } catch (const std::invalid_argument& e) {
        Logger::print("[Server] Interpret invalid argument {}, couldn't parse input: {}\n", e.what(), input);
        response_builder.add_error(e.what());
    } catch (const std::exception& e) {
        Logger::print("[Server] Interpret internal error, couldn't parse input {}\n", input);
        response_builder.add_error("Internal error");
    }

    auto response = response_builder.get_json_response();

    Logger::print("[Server] Interpret result {}\n", response);

    mg_rpc_ok(r, "%s", response.c_str());
    free(input);
}

static void solve(struct mg_rpc_req* r) {
    auto input = mg_json_get_str(r->frame, "$.params.input");

    if (input == NULL) {
        Logger::print("[Server] Solve couldn't find input\n", input);
        mg_rpc_err(r, 400, "Missing input");
        return;
    }

    Logger::print("[Server] Solve input {}\n", input);

    auto solver_process_manager = SolverProcessManager();
    auto result = solver_process_manager.try_solve(input);

    Logger::print("[Server] Solve result {}\n", result);

    mg_rpc_ok(r, "%s", result.c_str());

    free(input);
}

static void solve_with_steps(struct mg_rpc_req* r) {
    auto input = mg_json_get_str(r->frame, "$.params.input");

    if (input == NULL) {
        Logger::print("[Server] Solve with steps couldn't find input\n", input);
        mg_rpc_err(r, 400, "Missing input");
        return;
    }

    Logger::print("[Server] Solve with steps input {}\n", input);

    auto solver_process_manager = SolverProcessManager();
    auto result = solver_process_manager.try_solve(input);

    Logger::print("[Server] Solve with steps result {}\n", result);

    mg_rpc_ok(r, "%s", result.c_str());

    free(input);
}

// This RESTful server implements the following endpoints:
//   /websocket - upgrade to Websocket, and implement websocket echo server
void handler(struct mg_connection* c, int ev, void* ev_data, void* fn_data) {
    if (ev == MG_EV_OPEN) {
        // c->is_hexdumping = 1;
    }
    else if (ev == MG_EV_WS_OPEN) {
        c->label[0] = 'W'; // Mark this connection as an established WS client
    }
    else if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message* hm = (struct mg_http_message*)ev_data;
        if (mg_http_match_uri(hm, "/websocket")) {
            // Upgrade to websocket. From now on, a connection is a full-duplex
            // Websocket connection, which will receive MG_EV_WS_MSG events.
            mg_ws_upgrade(c, hm, NULL);
        }
        else {
            mg_http_reply(c, 404, "", "Not found");
        }
    }
    else if (ev == MG_EV_WS_MSG) {
        // Got websocket frame. Received data is wm->data
        struct mg_ws_message* wm = (struct mg_ws_message*)ev_data;
        struct mg_iobuf io = {0, 0, 0, 512};
        struct mg_rpc_req r = {&s_rpc_head, 0, mg_pfn_iobuf, &io, 0, wm->data};
        mg_rpc_process(&r);
        if (io.buf)
            mg_ws_send(c, (char*)io.buf, io.len, WEBSOCKET_OP_TEXT);
        mg_iobuf_free(&io);
    }
    (void)fn_data;
}

Server::Server(std::string listen_on, CachedParser cached_parser) :
    _listen_on(listen_on), _cached_parser(cached_parser) {
    mg_mgr_init(&_mgr);

    if constexpr (Consts::DEBUG) {
        mg_log_set(MG_LL_DEBUG);
    }

    mg_rpc_add(&s_rpc_head, mg_str("interpret"), interpret, NULL);
    mg_rpc_add(&s_rpc_head, mg_str("solve"), solve, NULL);
    mg_rpc_add(&s_rpc_head, mg_str("solve_with_steps"), solve_with_steps, NULL);
    mg_rpc_add(&s_rpc_head, mg_str("rpc.list"), mg_rpc_list, &s_rpc_head);

    global_cached_parser = &cached_parser;
}

void Server::run() {
    Logger::print("[Server] Starting on {}\n", _listen_on);

    mg_http_listen(&_mgr, _listen_on.c_str(), handler, NULL);
    for (;;) {
        mg_mgr_poll(&_mgr, 1000);
    }
}

Server::~Server() {
    mg_mgr_free(&_mgr);
    mg_rpc_del(&s_rpc_head, NULL);
}
