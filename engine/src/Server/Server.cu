#include "JsonFormatter.cuh"
#include "Parser/Parser.cuh"
#include "Server.cuh"
#include "Utils/CompileConstants.cuh"

#include <fmt/core.h>

// https://mongoose.ws/tutorials/json-rpc-over-websocket/

// Use of global variables is required, since mongoose callbacks are meant to be C functions.

static struct mg_rpc* s_rpc_head = NULL;
static CachedParser* global_cached_parser = NULL;
static Solver* global_solver = NULL;

template <typename... T> static void print_debug(fmt::format_string<T...> fmt, T&&... args) {
    if constexpr (Consts::DEBUG) {
        fmt::print(fmt, std::forward<T>(args)...);
    }
}

static void interpret(struct mg_rpc_req* r) {
    auto input = mg_json_get_str(r->frame, "$.params.input");

    if (input == NULL) {
        print_debug("[Server] Interpret couldn't find input\n");
        mg_rpc_err(r, 400, "Missing input");
        return;
    }

    print_debug("[Server] Interpret input {}\n", input);
    auto formatter = JsonFormatter();

    std::vector<std::string> errors;
    std::string json;

    try {
        auto parser_result = global_cached_parser->parse(input);
        json = formatter.format(&parser_result, nullptr, nullptr);
    } catch (const std::invalid_argument& e) {
        errors.push_back(e.what());

        json = formatter.format(nullptr, nullptr, &errors);
        print_debug("[Server] Interpret invalid argument {}, couldn't parse input: {}\n", e.what(), input);
    } catch (const std::exception& e) {
        errors.push_back("Internal error.");

        json = formatter.format(nullptr, nullptr, &errors);
        print_debug("[Server] Interpret internal error, couldn't parse input {}\n", input);
    }

    print_debug("[Server] Interpret result {}\n", json);

    mg_rpc_ok(r, "%s", json.c_str());
    free(input);
}

static void solve(struct mg_rpc_req* r) {
    auto input = mg_json_get_str(r->frame, "$.params.input");

    if (input == NULL) {
        print_debug("[Server] Solve couldn't find input\n", input);
        mg_rpc_err(r, 400, "Missing input");
        return;
    }

    print_debug("[Server] Solve input {}\n", input);

    auto parser_result = global_cached_parser->parse(input);

    print_debug("[Server] Parser result {}\n", parser_result.to_string());

    auto solver_result = global_solver->solve(parser_result);

    auto formatter = JsonFormatter();

    if (solver_result.has_value()) {
        auto json = formatter.format(&parser_result, &solver_result.value(), nullptr);
        print_debug("[Server] Solver found solution, returning response {}\n", json);
        mg_rpc_ok(r, "%s", json.c_str());
    }
    else {
        auto json = formatter.format(&parser_result, nullptr, nullptr);
        print_debug("[Server] Solver failed to find solution, returning response {}\n", json);
        mg_rpc_ok(r, "%s", json.c_str());
    }
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

Server::Server(std::string listen_on, CachedParser cached_parser, Solver solver) :
    _listen_on(listen_on), _cached_parser(cached_parser), _solver() {
    mg_mgr_init(&_mgr);

    if constexpr (Consts::DEBUG) {
        mg_log_set(MG_LL_DEBUG);
    }

    mg_rpc_add(&s_rpc_head, mg_str("interpret"), interpret, NULL);
    mg_rpc_add(&s_rpc_head, mg_str("solve"), solve, NULL);
    mg_rpc_add(&s_rpc_head, mg_str("rpc.list"), mg_rpc_list, &s_rpc_head);

    global_cached_parser = &cached_parser;
    global_solver = &solver;
}

void Server::run() {
    fmt::print("[Server] Starting on {}\n", _listen_on);

    mg_http_listen(&_mgr, _listen_on.c_str(), handler, NULL);
    for (;;) {
        mg_mgr_poll(&_mgr, 1000);
    }
}

Server::~Server() {
    mg_mgr_free(&_mgr);
    mg_rpc_del(&s_rpc_head, NULL);
}
