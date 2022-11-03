#include "Server.cuh"

// https://mongoose.ws/tutorials/json-rpc-over-websocket/

// Use of global variables is required, since mongoose callbacks are meant to be C functions.

static struct mg_rpc *s_rpc_head = NULL;


static void interpret(struct mg_rpc_req *r) {
    auto input = mg_json_get_str(r->frame, "$.params.input");

    if(input == NULL) {
        mg_rpc_err(r, 400, "Missing input");
        return;
    }

    mg_rpc_ok(r, "{\"inputInUtf\": \"%s\"}", input);
    free(input);
}

static void solve(struct mg_rpc_req *r) {
    auto input = mg_json_get_str(r->frame, "$.params.input");

    if(input == NULL) {
        mg_rpc_err(r, 400, "Missing input");
        return;
    }

    mg_rpc_ok(r, "{\"inputInUtf\": \"%s\"}", input);
    free(input);
}

// This RESTful server implements the following endpoints:
//   /websocket - upgrade to Websocket, and implement websocket echo server
void handler(struct mg_connection *c, int ev, void *ev_data, void *fn_data) {
    if (ev == MG_EV_OPEN) {
        // c->is_hexdumping = 1;
    } else if (ev == MG_EV_WS_OPEN) {
        c->label[0] = 'W';  // Mark this connection as an established WS client
    } else if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *) ev_data;
        if (mg_http_match_uri(hm, "/websocket")) {
            // Upgrade to websocket. From now on, a connection is a full-duplex
            // Websocket connection, which will receive MG_EV_WS_MSG events.
            mg_ws_upgrade(c, hm, NULL);
        } else {
            mg_http_reply(c, 404, "", "Not found");
        }
    } else if (ev == MG_EV_WS_MSG) {
        // Got websocket frame. Received data is wm->data
        struct mg_ws_message *wm = (struct mg_ws_message *) ev_data;
        struct mg_iobuf io = {0, 0, 0, 512};
        struct mg_rpc_req r = {&s_rpc_head, 0, mg_pfn_iobuf, &io, 0, wm->data};
        mg_rpc_process(&r);
        if (io.buf) mg_ws_send(c, (char *) io.buf, io.len, WEBSOCKET_OP_TEXT);
        mg_iobuf_free(&io);
    }
    (void) fn_data;
}

Server::Server(std::string listen_on) : listen_on(listen_on) {
    mg_mgr_init(&mgr);
    mg_log_set(MG_LL_DEBUG);

    mg_rpc_add(&s_rpc_head, mg_str("interpret"), interpret, NULL);
    mg_rpc_add(&s_rpc_head, mg_str("solve"), solve, NULL);
    mg_rpc_add(&s_rpc_head, mg_str("rpc.list"), mg_rpc_list, &s_rpc_head);
}

void Server::run() {
    fmt::print("Starting server on {}\n", listen_on);

    mg_http_listen(&mgr, listen_on.c_str(), handler, NULL);
    for (;;) {
        mg_mgr_poll(&mgr, 1000);
    }
}

Server::~Server() {
    mg_mgr_free(&mgr);
    mg_rpc_del(&s_rpc_head, NULL);
}
